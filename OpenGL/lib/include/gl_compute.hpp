#pragma once

#include <vector>

#include "amy_lab.hpp"

class GLCompute {
 public:
  static bool init();
  static void cleanup();
  static bool isAvailable();

  // Non-quantized calc_d: O(N^2), tiled GPU computation
  static std::vector<float> calcDNQ(const amy_lab* data, int N, int W, int H,
                                    float theta, float alpha);

  // Quantized calc_d: loops over quantized colors per pixel
  static std::vector<float> calcDQ(
      const amy_lab* data, int N, int W, int H,
      const std::vector<std::pair<amy_lab, int>>& qdata, float theta,
      float alpha);

  // Neighborhood calc_d with radius r
  static std::vector<float> rCalcD(const amy_lab* data, int W, int H, int r,
                                   float theta, float alpha);

  // Jacobi iterative solver (replaces Gauss-Seidel r_solve)
  static void rSolve(float* grayData, const float* d, int W, int H, int r,
                     int iterations);
};
