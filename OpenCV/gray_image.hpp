#pragma once

#include <opencv2/core/mat.hpp>

#include "color_image.hpp"

struct GrayImage {
  float *data;
  const int w, h, N;

  // this will shift our data to best match the
  // luminance channel of s.
  void post_solve(const ColorImage &s);

  explicit GrayImage(ColorImage &s);
  ~GrayImage();

  GrayImage(const GrayImage &) = delete;
  GrayImage &operator=(const GrayImage &) = delete;

  void complete_solve(const float *d);
  void r_solve(const float *d, int r);

  cv::Mat save(const char *fname) const;
  cv::Mat saveColor(const char *fname, const ColorImage &source) const;
};
