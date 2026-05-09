// Generates expected test values from the GPU implementation.
// Output is printed to stdout ready to paste into test_data.cpp.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "color_image.hpp"
#include "gray_image.hpp"
#include "test_data.hpp"
#include "utils.hpp"

static void print_vec(const std::vector<uint8_t>& v) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i % 15 == 0) std::cout << "      ";
    std::cout << static_cast<int>(v[i]);
    if (i + 1 < v.size()) std::cout << ", ";
    if ((i + 1) % 15 == 0) std::cout << '\n';
  }
  std::cout << '\n';
}

int main() {
  constexpr float alpha = 10.0f;
  constexpr bool quantize = false;
  constexpr float theta = 44.977226f * static_cast<float>(M_PI) / 180.0f;
  constexpr int mu = 10;
  constexpr int q_colors = 16;

  const auto test_input = get_test_image();
  constexpr int W = 1, H = 172;

  // ── mu_image ─────────────────────────────────────────────────────────────
  {
    ColorImage img(theta, alpha, false);
    img.load(test_input, W, H);
    GrayImage dest(img);
    auto d = img.r_calc_d(mu);
    dest.r_solve(d, mu);
    dest.post_solve(img);

    const auto gray = dest.save(nullptr);
    const auto colour = dest.saveColor(nullptr, img);

    std::cout << "namespace mu_image {\n\n";
    std::cout
        << "std::vector<uint8_t> get_expected_gray_image() {\n  return {\n";
    print_vec(gray);
    std::cout << "  };\n}\n\n";

    std::cout
        << "std::vector<uint8_t> get_expected_color_image() {\n  return {\n";
    print_vec(colour);
    std::cout << "  };\n}\n\n";
    std::cout << "}  // namespace mu_image\n\n";
  }

  // ── quantized_image ───────────────────────────────────────────────────────
  {
    ColorImage img(theta, alpha, true);
    img.load(test_input, W, H);
    GrayImage dest(img);

    auto d = img.r_calc_d(mu);
    dest.r_solve(d, mu);

    const auto quantized = quantify_image(test_input, q_colors);
    img.load_quant_data(quantized, W, H);
    dest.post_solve(img);

    const auto gray = dest.save(nullptr);
    const auto colour = dest.saveColor(nullptr, img);

    std::cout << "namespace quantized_image {\n\n";
    std::cout << "std::vector<uint8_t> get_expected_quantized_image() {\n  "
                 "return {\n";
    print_vec(quantized);
    std::cout << "  };\n}\n\n";

    std::cout
        << "std::vector<uint8_t> get_expected_gray_image() {\n  return {\n";
    print_vec(gray);
    std::cout << "  };\n}\n\n";

    std::cout
        << "std::vector<uint8_t> get_expected_color_image() {\n  return {\n";
    print_vec(colour);
    std::cout << "  };\n}\n\n";
    std::cout << "}  // namespace quantized_image\n";
  }

  return 0;
}
