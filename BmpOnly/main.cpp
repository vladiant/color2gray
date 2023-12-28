#include <args.h>

#include <charconv>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "bitmap.hpp"
#include "color_image.hpp"
#include "gray_image.hpp"
#include "time_bench.hpp"
#include "utils.hpp"

constexpr auto doc = R"doc(
color2gray algorithm BMP demo
Usage: color2gray image_name.bmp [options]
Options:
--help    print this message
--theta   Theta
--alpha   Alpha
--r       Mu
--q       Q
)doc";

constexpr float d2r = M_PI / 180.0;
constexpr float initial_theta = 3.14 / 6.0;

float parseString(std::string_view input) {
  // Expects the pattern identical to the one used by std::strtod in the default
  // ("C") locale
  float value{};
  const auto result = std::from_chars(input.begin(), input.end(), value);
  if (result.ec != std::errc()) {
    value = NAN;
  }
  return value;
}

int main(int argc, char** argv) {
  args::ArgParser parser(doc);

  parser.option("theta", std::to_string(initial_theta));
  parser.option("alpha", "10");
  parser.option("r", "0");
  parser.option("q");

  parser.parse(argc, argv);

  // Read image
  std::string image_name = "test.ppm";
  if (argc > 1) {
    image_name = argv[1];
  }

  int sourceWidth = 0;
  int sourceHeight = 0;
  const auto sourceData = readBMP(image_name, sourceWidth, sourceHeight);

  if (sourceData.empty()) {
    std::cout << "Failed to load file: " << image_name << '\n';
    return EXIT_FAILURE;
  }

  // Read parameters
  const auto theta_deg = parseString(parser.value("theta"));
  std::cout << "Theta = " << theta_deg << '\n';
  const float theta = theta_deg * d2r;

  const auto alpha = parseString(parser.value("alpha"));
  std::cout << "Alpha = " << alpha << '\n';

  const int r = std::stoi(parser.value("r"));
  std::cout << "mu = " << r << '\n';

  const auto q_value = parser.value("q");
  bool quantize = !q_value.empty();
  const int q_colors = quantize ? std::stoi(q_value) : 0;

  std::cout << "Executing color2gray algorithm on " << image_name
            << " with alpha=" << alpha << ", theta=" << theta_deg << '\n';

  const std::filesystem::path image_path{image_name};
  const std::string outname =
      image_path.stem().string() + "_c2g" + image_path.extension().string();
  const std::string outname_color = image_path.stem().string() + "_c2g_color" +
                                    image_path.extension().string();

  // load the image
  ColorImage initial_image(theta, alpha, quantize);
  initial_image.load(sourceData, sourceWidth, sourceHeight);
  GrayImage dest(initial_image);

  const auto start = std::chrono::high_resolution_clock::now();

  // solve, either using the complete case or the neighboorhod case.
  if (r) {
    std::vector<float> d;
    {
      const TimeBench bench{"r_calc_d"};
      d = initial_image.r_calc_d(r);
    }

    const TimeBench bench{"r_solve"};
    dest.r_solve(d, r);
  } else {
    if (quantize) {
      std::cout << "q = " << q_colors << '\n';
      const TimeBench bench{"Create a quantized image"};
      const auto quantized = quantify_image(sourceData, q_colors);
      initial_image.load_quant_data(quantized, sourceWidth, sourceHeight);
    }

    std::vector<float> d;
    {
      const TimeBench bench{"calc_d"};
      d = initial_image.calc_d();
    }

    const TimeBench bench{"complete_solve"};
    dest.complete_solve(d);
  }

  {
    const TimeBench bench{"post_solve"};
    dest.post_solve(initial_image);
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto process_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "c2g completed in " << process_time << " milliseconds\n";

  dest.save(outname.c_str());
  dest.saveColor(outname_color.c_str(), initial_image);

  return EXIT_SUCCESS;
}
