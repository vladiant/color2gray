#pragma once

#include <utility>
#include <vector>

#include "amy_lab.hpp"

// ---------------------------------------------------------------------------
// CUDA-accelerated implementations of the color2gray computational kernels.
//
// Note on r_calc_d:
//   The CPU reference implementation uses a symmetric double-counting loop
//   (each ordered pair (i,j) is visited twice), so it produces
//   d[i] = 2 * Σ_{j∈nbhd(i)} Δ(i,j).  The CUDA wrapper replicates this
//   factor-of-2 so results are numerically consistent with the CPU path.
//
// Note on r_solve:
//   The CPU uses Gauss-Seidel relaxation (sequential updates within a sweep).
//   The CUDA implementation uses Jacobi iteration (each sweep reads from the
//   previous step) with 2× the iteration count to reach equivalent quality.
// ---------------------------------------------------------------------------

/// Full O(N²) delta computation on GPU.
std::vector<float> cuda_calc_d_nq(int N, const amy_lab* data, float theta,
                                  float alpha);

/// Quantized delta computation on GPU.
std::vector<float> cuda_calc_d_q(
    int N, const amy_lab* data,
    const std::vector<std::pair<amy_lab, int>>& qdata, float theta,
    float alpha);

/// Neighbourhood delta computation on GPU.
std::vector<float> cuda_r_calc_d(int W, int H, const amy_lab* data, float theta,
                                 float alpha, int r);

/// Parallel closed-form complete-solve on GPU.
/// Equivalent to the sequential recurrence:
///   grayData[i] = grayData[0] + (d[i] - d[0]) / N
void cuda_complete_solve(int N, float* grayData, const float* d);

/// Jacobi-iteration neighbourhood solve on GPU.
/// Uses Red-Black Gauss-Seidel (30 sweeps = 60 kernel launches) which
/// converges at the same rate as the CPU sequential Gauss-Seidel.
void cuda_r_solve(int W, int H, int r, float* grayData, const float* d);

/// Adjust grey values to match mean luminance of the source image (GPU).
void cuda_post_solve(int N, float* grayData, const float* labL);
