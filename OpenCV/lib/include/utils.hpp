#pragma once

#include <opencv2/core/mat.hpp>

inline float clamp(float x, float x_min, float x_max) {
  if (x < x_min)
    return (x_min);
  else if (x > x_max)
    return (x_max);
  else
    return (x);
}

inline float sq(float s) { return s * s; }

cv::Mat3b quantify_image(const cv::Mat3b& source, int q_colors);
