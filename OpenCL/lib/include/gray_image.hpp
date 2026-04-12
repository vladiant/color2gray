#pragma once

#include <vector>

#include "color_image.hpp"

#ifdef HAS_OPENCL
class OpenCLContext;
#endif

class GrayImage {
  std::vector<float> mData;
  const int mW, mH, mN;

 public:
  // this will shift our data to best match the
  // luminance channel of s.
  void post_solve(const ColorImage &s);

  explicit GrayImage(const ColorImage &s);

  GrayImage(const GrayImage &) = delete;
  GrayImage &operator=(const GrayImage &) = delete;

  void complete_solve(const std::vector<float> &d);
  void r_solve(const std::vector<float> &d, int r);

#ifdef HAS_OPENCL
  /// GPU-accelerated Jacobi solver (replaces Gauss-Seidel r_solve).
  void r_solve_ocl(const std::vector<float> &d, int r, OpenCLContext &ocl);
#endif

  std::vector<uint8_t> save(const char *fname) const;
  std::vector<uint8_t> saveColor(const char *fname,
                                 const ColorImage &source) const;
};
