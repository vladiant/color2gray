#pragma once

#include <cstdint>
#include <vector>

std::vector<uint8_t> get_test_image();

namespace full_image {
std::vector<uint8_t> get_expected_gray_image();

std::vector<uint8_t> get_expected_color_image();
}  // namespace full_image

namespace mu_image {
std::vector<uint8_t> get_expected_gray_image();

std::vector<uint8_t> get_expected_color_image();
}  // namespace mu_image

namespace quantized_image {

std::vector<uint8_t> get_expected_quantized_image();

std::vector<uint8_t> get_expected_gray_image();

std::vector<uint8_t> get_expected_color_image();
}  // namespace quantized_image
