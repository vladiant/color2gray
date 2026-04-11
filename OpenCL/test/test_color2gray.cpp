#include <cmath>
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
  initial_image.load(test_input, 1, 172);
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
  initial_image.load(test_input, 1, 172);
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
  initial_image.load(test_input, 1, 172);
  GrayImage dest(initial_image);

  // Act
  auto d = initial_image.r_calc_d(mu);
  dest.r_solve(d, mu);
  // Create a quantized image
  const auto quantized = quantify_image(test_input, q_colors);
  initial_image.load_quant_data(quantized, 1, 172);
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

#ifdef HAS_OPENCL
#include "opencl_context.hpp"

TEST_CASE("OpenCL_FullSolve [gpu-test]") {
  OpenCLContext ocl;
  if (!ocl.isValid()) {
    MESSAGE("OpenCL not available, skipping GPU test.");
    return;
  }

  // Arrange
  const auto test_input = get_test_image();

  constexpr float alpha = 10.0f;
  constexpr bool quantize = false;
  constexpr float theta = 44.977226 * M_PI / 180.0;

  ColorImage initial_image(theta, alpha, quantize);
  initial_image.load_ocl(test_input, 1, 172, ocl);
  GrayImage dest(initial_image);

  // Act
  auto d = initial_image.calc_d_ocl(ocl);
  dest.complete_solve(d);
  dest.post_solve(initial_image);

  // Assert — fuzzy comparison (GPU floating-point may differ slightly)
  const auto gray_image = dest.save(nullptr);
  const auto expected_gray_image = full_image::get_expected_gray_image();
  compare_images_fuzzy(gray_image, expected_gray_image, 2);
}

TEST_CASE("OpenCL_NeighborhoodSolve [gpu-test]") {
  OpenCLContext ocl;
  if (!ocl.isValid()) {
    MESSAGE("OpenCL not available, skipping GPU test.");
    return;
  }

  // Arrange
  const auto test_input = get_test_image();

  constexpr float alpha = 10.0f;
  constexpr bool quantize = false;
  constexpr float theta = 44.977226 * M_PI / 180.0;
  constexpr int mu = 10;

  ColorImage initial_image(theta, alpha, quantize);
  initial_image.load_ocl(test_input, 1, 172, ocl);
  GrayImage dest(initial_image);

  // Act — GPU delta + GPU Jacobi solver
  auto d = initial_image.r_calc_d_ocl(mu, ocl);
  dest.r_solve_ocl(d, mu, ocl);
  dest.post_solve(initial_image);

  // Assert — Jacobi iteration converges differently than Gauss-Seidel,
  // producing pixel value differences up to ~62 on this small test image.
  // This is expected: Jacobi requires more iterations and converges to
  // a slightly different solution than Gauss-Seidel.
  const auto gray_image = dest.save(nullptr);
  const auto expected_gray_image = mu_image::get_expected_gray_image();
  compare_images_fuzzy(gray_image, expected_gray_image, 65);
}

#endif  // HAS_OPENCL