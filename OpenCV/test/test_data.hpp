#pragma once

#include <opencv2/core.hpp>

cv::Mat get_test_image();

namespace full_image {
cv::Mat get_expected_gray_image();

cv::Mat get_expected_color_image();
}  // namespace full_image

namespace mu_image {
cv::Mat get_expected_gray_image();

cv::Mat get_expected_color_image();
}  // namespace mu_image
