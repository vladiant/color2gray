#include "test_utils.hpp"

#include <cmath>

#include "doctest/doctest.h"

void compare_images(const std::vector<uint8_t>& lhs,
                    const std::vector<uint8_t>& rhs) {
  REQUIRE(lhs.size() == rhs.size());

  auto it_rhs = rhs.begin();
  auto it_lhs = lhs.begin();
  std::size_t index = 0;
  for (; it_rhs != rhs.end(); ++it_rhs, ++it_lhs, index++) {
    CHECK_MESSAGE(*it_rhs == *it_lhs, index);
  }
}

void compare_images_fuzzy(const std::vector<uint8_t>& lhs,
                          const std::vector<uint8_t>& rhs, int tolerance) {
  REQUIRE(lhs.size() == rhs.size());

  auto it_rhs = rhs.begin();
  auto it_lhs = lhs.begin();
  std::size_t index = 0;
  for (; it_rhs != rhs.end(); ++it_rhs, ++it_lhs, index++) {
    const int diff = std::abs(static_cast<int>(*it_rhs) - static_cast<int>(*it_lhs));
    CHECK_MESSAGE(diff <= tolerance, index);
  }
}