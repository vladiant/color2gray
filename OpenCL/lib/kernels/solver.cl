// solver.cl — Jacobi iteration kernel for r_solve
// Replaces Gauss-Seidel with Jacobi for GPU parallelism.
// Each work-item updates one pixel by averaging its r-radius neighborhood.
// Host ping-pongs between src and dst buffers each iteration.

__kernel void jacobi_iteration(__global const float* d,
                               __global const float* src,
                               __global float* dst,
                               const int W,
                               const int H,
                               const int r) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= W || y >= H) return;

  float sum = 0.0f;
  int count = 0;

  const int xx_min = max(x - r, 0);
  const int xx_max = min(x + r, W - 1);
  const int yy_min = max(y - r, 0);
  const int yy_max = min(y + r, H - 1);

  for (int xx = xx_min; xx <= xx_max; xx++) {
    for (int yy = yy_min; yy <= yy_max; yy++) {
      sum += src[xx + yy * W];
      count++;
    }
  }

  dst[x + y * W] = (d[x + W * y] + sum) / (float)count;
}
