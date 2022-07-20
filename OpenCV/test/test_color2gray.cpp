#include "color_image.hpp"
#include "doctest/doctest.h"
#include "gray_image.hpp"
#include "test_utils.hpp"
#include "utils.hpp"

TEST_CASE("InputImage_TransformedImages [full-test]") {
  // Arrange
  const auto test_input = get_test_image();

  constexpr float alpha = 10.0f;
  constexpr bool quantize = false;
  constexpr float theta = 44.977226 * M_PI / 180.0;

  ColorImage initial_image(theta, alpha, quantize);
  initial_image.load(test_input);
  GrayImage dest(initial_image);

  // Act
  auto d = initial_image.calc_d();
  dest.complete_solve(d);
  dest.post_solve(initial_image);

  // Assert
  const auto gray_image = dest.save(nullptr);
  const auto expected_gray_image = get_expected_gray_image();
  REQUIRE(expected_gray_image.size() == gray_image.size());

  auto it_gray_test = gray_image.begin<cv::Vec3b>();
  auto it_gray_expected = expected_gray_image.begin<cv::Vec3b>();
  for (; it_gray_test != gray_image.end<cv::Vec3b>();
       ++it_gray_test, ++it_gray_expected) {
    REQUIRE(*it_gray_test == *it_gray_expected);
  }

  const auto color_image = dest.saveColor(nullptr, initial_image);
  const auto expected_color_image = get_expected_color_image();
  REQUIRE(expected_color_image.size() == color_image.size());

  auto it_color_test = color_image.begin<cv::Vec3b>();
  auto it_color_expected = expected_color_image.begin<cv::Vec3b>();
  for (; it_color_test != color_image.end<cv::Vec3b>();
       ++it_color_test, ++it_color_expected) {
    REQUIRE(*it_color_test == *it_color_expected);
  }
}