// calc_delta.cl — Delta computation kernels for color2gray
// Inlines the calc_delta logic from utils.cpp

// Helper: crunch function (saturation weighting)
float crunch_f(float chromDist, float alpha) {
  return (alpha == 0.0f) ? 0.0f : alpha * tanh(chromDist / alpha);
}

// Helper: compute delta between two LAB pixels
float calc_delta_f(float l_i, float a_i, float b_i,
                   float l_j, float a_j, float b_j,
                   float theta, float alpha) {
  const float dL = l_i - l_j;
  const float da = a_i - a_j;
  const float db = b_i - b_j;
  const float dC = crunch_f(sqrt(da * da + db * db), alpha);

  if (fabs(dL) > dC) {
    return dL;
  }

  return dC * ((cos(theta) * da + sin(theta) * db) > 0.0f ? 1.0f : -1.0f);
}

// ============================================================
// Kernel: calc_d_nq — all-pairs delta (full solve, no quantization)
// Each work-item computes d[i] = sum_j calc_delta(i, j) for all j != i.
// Uses the "obvious" O(N) per-pixel formulation for trivial parallelism.
// ============================================================
__kernel void calc_d_nq_kernel(__global const float* lab_data,
                               __global float* d,
                               const float theta,
                               const float alpha,
                               const int N) {
  const int i = get_global_id(0);
  if (i >= N) return;

  const float l_i = lab_data[3 * i];
  const float a_i = lab_data[3 * i + 1];
  const float b_i = lab_data[3 * i + 2];

  float sum = 0.0f;
  for (int j = 0; j < N; j++) {
    if (j == i) continue;
    const float l_j = lab_data[3 * j];
    const float a_j = lab_data[3 * j + 1];
    const float b_j = lab_data[3 * j + 2];
    sum += calc_delta_f(l_i, a_i, b_i, l_j, a_j, b_j, theta, alpha);
  }

  d[i] = sum;
}

// ============================================================
// Kernel: r_calc_d — neighborhood delta (radius r)
// Each work-item computes d[i] = sum over neighborhood of calc_delta(i, j).
// 2D NDRange: global size = (W, H)
// ============================================================
__kernel void r_calc_d_kernel(__global const float* lab_data,
                              __global float* d,
                              const float theta,
                              const float alpha,
                              const int W,
                              const int H,
                              const int r) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= W || y >= H) return;

  const int i = x + y * W;
  const float l_i = lab_data[3 * i];
  const float a_i = lab_data[3 * i + 1];
  const float b_i = lab_data[3 * i + 2];

  float sum = 0.0f;

  const int xx_min = max(x - r, 0);
  const int xx_max = min(x + r, W - 1);
  const int yy_min = max(y - r, 0);
  const int yy_max = min(y + r, H - 1);

  for (int xx = xx_min; xx <= xx_max; xx++) {
    for (int yy = yy_min; yy <= yy_max; yy++) {
      const int j = xx + yy * W;
      const float l_j = lab_data[3 * j];
      const float a_j = lab_data[3 * j + 1];
      const float b_j = lab_data[3 * j + 2];
      sum += calc_delta_f(l_i, a_i, b_i, l_j, a_j, b_j, theta, alpha);
    }
  }

  d[i] = sum;
}

// ============================================================
// Kernel: calc_d_q — quantized delta
// Each work-item computes d[i] using quantized palette colors.
// ============================================================
__kernel void calc_d_q_kernel(__global const float* lab_data,
                              __global const float* qlab_data,
                              __global const int* qcounts,
                              __global float* d,
                              const float theta,
                              const float alpha,
                              const int N,
                              const int Q) {
  const int i = get_global_id(0);
  if (i >= N) return;

  const float l_i = lab_data[3 * i];
  const float a_i = lab_data[3 * i + 1];
  const float b_i = lab_data[3 * i + 2];

  float sum = 0.0f;
  for (int p = 0; p < Q; p++) {
    const float l_p = qlab_data[3 * p];
    const float a_p = qlab_data[3 * p + 1];
    const float b_p = qlab_data[3 * p + 2];
    const int count = qcounts[p];

    const float dL = l_i - l_p;
    const float da = a_i - a_p;
    const float db = b_i - b_p;
    const float dC = crunch_f(sqrt(da * da + db * db), alpha);

    float delta;
    if (fabs(dL) > dC) {
      delta = (float)count * dL;
    } else {
      delta = (float)count * dC *
              ((cos(theta) * da + sin(theta) * db) > 0.0f ? 1.0f : -1.0f);
    }
    sum += delta;
  }

  d[i] = sum;
}
