#include <iostream>

#include "color_image.hpp"
#include "doctest/doctest.h"
#include "gray_image.hpp"
#include "test_data.hpp"
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
  const auto expected_gray_image = full_image::get_expected_gray_image();
  compare_images(gray_image, expected_gray_image);

  const auto color_image = dest.saveColor(nullptr, initial_image);
  const auto expected_color_image = full_image::get_expected_color_image();
  compare_images(color_image, expected_color_image);
}

TEST_CASE("InputImage_Mu1_TransformedImages [full-test]") {
  // Arrange
  const auto test_input = get_test_image();

  constexpr float alpha = 10.0f;
  constexpr bool quantize = false;
  constexpr float theta = 44.977226 * M_PI / 180.0;
  constexpr int mu = 10;

  ColorImage initial_image(theta, alpha, quantize);
  initial_image.load(test_input);
  GrayImage dest(initial_image);

  // Act
  auto d = initial_image.r_calc_d(mu);
  dest.r_solve(d, mu);
  dest.post_solve(initial_image);

  // Assert
  const auto gray_image = dest.save(nullptr);
  const auto expected_gray_image = mu_image::get_expected_gray_image();
  compare_images(gray_image, expected_gray_image);

  const auto color_image = dest.saveColor(nullptr, initial_image);
  const auto expected_color_image = mu_image::get_expected_color_image();
  compare_images(color_image, expected_color_image);
}

TEST_CASE("InputImage_Mu1Quantize_TransformedImages [full-test]") {
  // Arrange
  const auto test_input = get_test_image();

  constexpr float alpha = 10.0f;
  constexpr bool quantize = true;
  constexpr float theta = 44.977226 * M_PI / 180.0;
  constexpr int mu = 10;
  constexpr int q_colors = 16;

  ColorImage initial_image(theta, alpha, quantize);
  initial_image.load(test_input);
  GrayImage dest(initial_image);

  // Act
  auto d = initial_image.r_calc_d(mu);
  dest.r_solve(d, mu);
  // Create a quantized image
  const auto quantized = quantify_image(test_input, q_colors);
  initial_image.load_quant_data(quantized);
  dest.post_solve(initial_image);

  // Assert
  const auto expected_quantized =
      quantized_image::get_expected_quantized_image();
  compare_images(quantized, expected_quantized);

  const auto gray_image = dest.save(nullptr);
  const auto expected_gray_image = quantized_image::get_expected_gray_image();
  compare_images(gray_image, expected_gray_image);

  const auto color_image = dest.saveColor(nullptr, initial_image);
  const auto expected_color_image = quantized_image::get_expected_color_image();
  compare_images(color_image, expected_color_image);
}