#include "cuda_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Error-checking helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d – %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(_err));                                        \
      throw std::runtime_error(cudaGetErrorString(_err));                      \
    }                                                                          \
  } while (0)

static constexpr int kBlock = 256;

// ---------------------------------------------------------------------------
// Device helper functions
// ---------------------------------------------------------------------------

__device__ __forceinline__ float d_crunch(float chromDist, float alpha) {
  return (alpha == 0.0f) ? 0.0f : alpha * tanhf(chromDist / alpha);
}

// Compute signed perceptual difference Δ(pixel_a, pixel_b).
__device__ __forceinline__ float d_delta(float al, float aa, float ab,
                                          float bl, float ba, float bb,
                                          float theta, float alpha) {
  const float dL = al - bl;
  const float da = aa - ba;
  const float db = ab - bb;
  const float dC = d_crunch(sqrtf(da * da + db * db), alpha);
  if (fabsf(dL) > dC) return dL;
  return dC * ((__cosf(theta) * da + __sinf(theta) * db) > 0.0f ? 1.0f : -1.0f);
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Full O(N²) delta – each thread owns one pixel i and sums over all j.
// The CPU reference uses an antisymmetric pair-loop which gives
// d[i] = Σ_{j≠i} Δ(i,j) – identical to what this kernel computes since
// the self-pair contributes 0.
__global__ void k_calc_d_nq(int N,
                              const float* __restrict__ L,
                              const float* __restrict__ A,
                              const float* __restrict__ B,
                              float theta, float alpha,
                              float* __restrict__ d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  const float li = L[i], ai = A[i], bi = B[i];
  float sum = 0.0f;
  for (int j = 0; j < N; ++j)
    sum += d_delta(li, ai, bi, L[j], A[j], B[j], theta, alpha);
  d[i] = sum;  // j==i contributes 0, safe to include
}

// Quantised delta – each thread owns one pixel i, sums over Q cluster centres.
// Replicates calc_qdelta() which weights by qCount.
__global__ void k_calc_d_q(int N, int Q,
                             const float* __restrict__ L,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             const float* __restrict__ qL,
                             const float* __restrict__ qA,
                             const float* __restrict__ qB,
                             const int* __restrict__   qCnt,
                             float theta, float alpha,
                             float* __restrict__ d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  const float li = L[i], ai = A[i], bi = B[i];
  float sum = 0.0f;
  for (int p = 0; p < Q; ++p) {
    const float dL  = li - qL[p];
    const float da  = ai - qA[p];
    const float db  = bi - qB[p];
    const float dC  = d_crunch(sqrtf(da * da + db * db), alpha);
    float delta;
    if (fabsf(dL) > dC) {
      delta = static_cast<float>(qCnt[p]) * dL;
    } else {
      const float sign =
          (__cosf(theta) * da + __sinf(theta) * db) > 0.0f ? 1.0f : -1.0f;
      delta = static_cast<float>(qCnt[p]) * dC * sign;
    }
    sum += delta;
  }
  d[i] = sum;
}

// Neighbourhood delta – each thread handles one pixel.
// The CPU implementation processes each ordered pair (i,j) twice (symmetric
// double-loop), giving d[i] = 2·Σ_{j∈nbhd(i)} Δ(i,j).  We replicate the
// factor-of-2 to produce an identical result.
__global__ void k_r_calc_d(int W, int H,
                             const float* __restrict__ L,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             float theta, float alpha, int r,
                             float* __restrict__ d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= W * H) return;

  const int x = idx % W;
  const int y = idx / W;
  const float li = L[idx], ai = A[idx], bi = B[idx];
  float sum = 0.0f;

  for (int xx = max(0, x - r), xx_end = min(W - 1, x + r); xx <= xx_end; ++xx)
    for (int yy = max(0, y - r), yy_end = min(H - 1, y + r); yy <= yy_end; ++yy)
      sum += d_delta(li, ai, bi, L[xx + yy * W], A[xx + yy * W], B[xx + yy * W],
                     theta, alpha);

  d[idx] = 2.0f * sum;  // match CPU double-counting
}

// Parallel closed-form solution: grayData[i] = gray0 + (d[i] - d0) / N
// (algebraically equivalent to the sequential recurrence for i ≥ 1).
__global__ void k_complete_solve(int N, float* grayData, const float* d,
                                  float gray0, float d0) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0 || i >= N) return;
  grayData[i] = gray0 + (d[i] - d0) / static_cast<float>(N);
}

// One Red-Black Gauss-Seidel phase.
// phase == 0 → update pixels where (x + y) % 2 == 0 ("red")
// phase == 1 → update pixels where (x + y) % 2 == 1 ("black")
// Pixels of the other colour are left unchanged (copied as-is).
// Together, two phases constitute one full Gauss-Seidel sweep and converge
// at the same rate as the sequential row-by-row sweep.
__global__ void k_r_solve_rbgs(int W, int H,
                                const float* __restrict__ rhs,
                                const float* __restrict__ gIn,
                                float* __restrict__ gOut,
                                int r, int phase) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= W * H) return;

  const int x = idx % W;
  const int y = idx / W;

  if ((x + y) % 2 != phase) {
    // Not our colour this phase – propagate unchanged.
    gOut[idx] = gIn[idx];
    return;
  }

  float sum = 0.0f;
  int   cnt = 0;

  for (int xx = max(0, x - r), xx_end = min(W - 1, x + r); xx <= xx_end; ++xx)
    for (int yy = max(0, y - r), yy_end = min(H - 1, y + r); yy <= yy_end; ++yy) {
      sum += gIn[xx + yy * W];
      ++cnt;
    }

  gOut[idx] = (rhs[idx] + sum) / static_cast<float>(cnt);
}

// Partial-sum reduction: Σ(grayData[i] - labL[i])
__global__ void k_reduce_error(int N,
                                const float* __restrict__ grayData,
                                const float* __restrict__ labL,
                                float* __restrict__ partialSums) {
  extern __shared__ float sdata[];
  const int tid = threadIdx.x;
  const int i   = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < N) ? (grayData[i] - labL[i]) : 0.0f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) partialSums[blockIdx.x] = sdata[0];
}

// Subtract a scalar error from every element.
__global__ void k_post_solve_apply(int N, float* grayData, float error) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  grayData[i] -= error;
}

// ---------------------------------------------------------------------------
// Internal helper: unpack amy_lab AoS → three SoA device arrays
// ---------------------------------------------------------------------------
static void upload_lab(int N, const amy_lab* data,
                        float** d_L, float** d_A, float** d_B) {
  std::vector<float> hL(N), hA(N), hB(N);
  for (int i = 0; i < N; ++i) {
    hL[i] = data[i].l;
    hA[i] = data[i].a;
    hB[i] = data[i].b;
  }
  CUDA_CHECK(cudaMalloc(d_L, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(d_A, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(d_B, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(*d_L, hL.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(*d_A, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(*d_B, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// Public C++ wrappers
// ---------------------------------------------------------------------------

std::vector<float> cuda_calc_d_nq(int N, const amy_lab* data,
                                   float theta, float alpha) {
  float *dL, *dA, *dB, *dd;
  upload_lab(N, data, &dL, &dA, &dB);
  CUDA_CHECK(cudaMalloc(&dd, N * sizeof(float)));

  const int grid = (N + kBlock - 1) / kBlock;
  k_calc_d_nq<<<grid, kBlock>>>(N, dL, dA, dB, theta, alpha, dd);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> result(N);
  CUDA_CHECK(cudaMemcpy(result.data(), dd, N * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(dL); cudaFree(dA); cudaFree(dB); cudaFree(dd);
  return result;
}

std::vector<float> cuda_calc_d_q(
    int N, const amy_lab* data,
    const std::vector<std::pair<amy_lab, int>>& qdata,
    float theta, float alpha) {
  const int Q = static_cast<int>(qdata.size());

  float *dL, *dA, *dB;
  upload_lab(N, data, &dL, &dA, &dB);

  std::vector<float> hqL(Q), hqA(Q), hqB(Q);
  std::vector<int>   hqC(Q);
  for (int p = 0; p < Q; ++p) {
    hqL[p] = qdata[p].first.l;
    hqA[p] = qdata[p].first.a;
    hqB[p] = qdata[p].first.b;
    hqC[p] = qdata[p].second;
  }

  float *dqL, *dqA, *dqB, *dd;
  int   *dqC;
  CUDA_CHECK(cudaMalloc(&dqL, Q * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dqA, Q * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dqB, Q * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dqC, Q * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dd,  N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dqL, hqL.data(), Q * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dqA, hqA.data(), Q * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dqB, hqB.data(), Q * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dqC, hqC.data(), Q * sizeof(int),   cudaMemcpyHostToDevice));

  const int grid = (N + kBlock - 1) / kBlock;
  k_calc_d_q<<<grid, kBlock>>>(N, Q, dL, dA, dB, dqL, dqA, dqB, dqC,
                                theta, alpha, dd);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> result(N);
  CUDA_CHECK(cudaMemcpy(result.data(), dd, N * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(dL);  cudaFree(dA);  cudaFree(dB);
  cudaFree(dqL); cudaFree(dqA); cudaFree(dqB); cudaFree(dqC);
  cudaFree(dd);
  return result;
}

std::vector<float> cuda_r_calc_d(int W, int H, const amy_lab* data,
                                  float theta, float alpha, int r) {
  const int N = W * H;
  float *dL, *dA, *dB, *dd;
  upload_lab(N, data, &dL, &dA, &dB);
  CUDA_CHECK(cudaMalloc(&dd, N * sizeof(float)));

  const int grid = (N + kBlock - 1) / kBlock;
  k_r_calc_d<<<grid, kBlock>>>(W, H, dL, dA, dB, theta, alpha, r, dd);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> result(N);
  CUDA_CHECK(cudaMemcpy(result.data(), dd, N * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(dL); cudaFree(dA); cudaFree(dB); cudaFree(dd);
  return result;
}

void cuda_complete_solve(int N, float* grayData, const float* d) {
  float *dGray, *dD;
  CUDA_CHECK(cudaMalloc(&dGray, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dD,    N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dGray, grayData, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dD,    d,        N * sizeof(float), cudaMemcpyHostToDevice));

  const float gray0 = grayData[0];
  const float d0    = d[0];

  const int grid = (N + kBlock - 1) / kBlock;
  k_complete_solve<<<grid, kBlock>>>(N, dGray, dD, gray0, d0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(grayData, dGray, N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(dGray); cudaFree(dD);
}

void cuda_r_solve(int W, int H, int r, float* grayData, const float* d) {
  // Two phases (red + black) per Gauss-Seidel sweep.
  // 30 full sweeps matches the CPU sequential implementation.
  constexpr int kSweeps = 30;
  const int N = W * H;

  float *dA, *dB, *dD;  // ping-pong buffers
  CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dD, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, grayData, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dD, d,        N * sizeof(float), cudaMemcpyHostToDevice));

  const int grid = (N + kBlock - 1) / kBlock;
  // Track which buffer is current (starts at A).
  float* cur  = dA;
  float* next = dB;
  for (int sweep = 0; sweep < kSweeps; ++sweep) {
    // Phase 0 (red)
    k_r_solve_rbgs<<<grid, kBlock>>>(W, H, dD, cur, next, r, 0);
    CUDA_CHECK(cudaGetLastError());
    std::swap(cur, next);
    // Phase 1 (black) – reads the updated red pixels
    k_r_solve_rbgs<<<grid, kBlock>>>(W, H, dD, cur, next, r, 1);
    CUDA_CHECK(cudaGetLastError());
    std::swap(cur, next);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(grayData, cur, N * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(dA); cudaFree(dB); cudaFree(dD);
}

void cuda_post_solve(int N, float* grayData, const float* labL) {
  const int grid = (N + kBlock - 1) / kBlock;

  float *dGray, *dLabL, *dPartial;
  CUDA_CHECK(cudaMalloc(&dGray,    N    * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dLabL,    N    * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dPartial, grid * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dGray, grayData, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dLabL, labL,     N * sizeof(float), cudaMemcpyHostToDevice));

  // Parallel reduction to compute Σ(grayData[i] - labL[i])
  k_reduce_error<<<grid, kBlock, kBlock * sizeof(float)>>>(N, dGray, dLabL, dPartial);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hPartial(grid);
  CUDA_CHECK(cudaMemcpy(hPartial.data(), dPartial, grid * sizeof(float), cudaMemcpyDeviceToHost));
  float error = 0.0f;
  for (int i = 0; i < grid; ++i) error += hPartial[i];
  error /= static_cast<float>(N);

  k_post_solve_apply<<<grid, kBlock>>>(N, dGray, error);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(grayData, dGray, N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(dGray); cudaFree(dLabL); cudaFree(dPartial);
}
