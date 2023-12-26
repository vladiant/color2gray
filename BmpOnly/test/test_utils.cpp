#include "test_utils.hpp"

#include "doctest/doctest.h"

void compare_images(const cv::Mat3b& lhs, const cv::Mat3b& rhs) {
  REQUIRE(lhs.size() == rhs.size());

  auto it_rhs = rhs.begin();
  auto it_lhs = lhs.begin();
  for (; it_rhs != rhs.end(); ++it_rhs, ++it_lhs) {
    REQUIRE(*it_rhs == *it_lhs);
  }
}