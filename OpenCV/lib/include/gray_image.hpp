#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>

#include "color_image.hpp"

class GrayImage {
  std::vector<float> mData;
  const int mW, mH, mN;

 public:
  // this will shift our data to best match the
  // luminance channel of s.
  void post_solve(const ColorImage &s);

  explicit GrayImage(ColorImage &s);

  GrayImage(const GrayImage &) = delete;
  GrayImage &operator=(const GrayImage &) = delete;

  void complete_solve(const std::vector<float> &d);
  void r_solve(const std::vector<float> &d, int r);

  cv::Mat save(const char *fname) const;
  cv::Mat saveColor(const char *fname, const ColorImage &source) const;
};
