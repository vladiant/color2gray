#include <cmath>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "color_image.hpp"
#include "gray_image.hpp"
#include "time_bench.hpp"
#include "utils.hpp"
#include "bitmap.hpp"

constexpr auto doc = R"doc(
color2gray algorithm BMP demo
)doc";

constexpr float d2r = M_PI / 180.0;
constexpr float initial_theta = 3.14 / 6.0;

int main(int argc, char** argv) {
  cv::String keys =
      "{help h usage ? |     | print this message   }"
      "{theta          | " +
      std::to_string(initial_theta) +
      "  | Theta  }"
      "{alpha          | 10.0  | Alpha   }"
      "{r              |  0  | Mu }"
      "{q              |     | Q }";
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about(doc);

  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  // Read image
  std::string image_name = "test.ppm";
  if (argc > 1) {
    image_name = argv[1];
  }

  int sourceWidth;
  int sourceHeight;
  const auto sourceData = readBMP(image_name, sourceWidth, sourceHeight);

  if (sourceData.empty()) {
    std::cout << "Failed to load file: " << image_name << '\n';
    return EXIT_FAILURE;
  }

  // Read parameters
  const auto theta_deg = parser.get<float>("theta");
  std::cout << "Theta = " << theta_deg << '\n';
  const float theta = theta_deg * d2r;

  const auto alpha = parser.get<float>("alpha");
  std::cout << "Alpha = " << alpha << '\n';

  const int r = parser.get<int>("r");
  std::cout << "mu = " << r << '\n';

  bool quantize = false;
  int q_colors = parser.get<int>("q");
  if (parser.has("q")) {
    quantize = true;
    std::cout << "q = " << q_colors << '\n';
  }

  std::cout << "Executing color2gray algorithm on " << image_name
            << " with alpha=" << alpha << ", theta=" << theta_deg << '\n';

  const std::filesystem::path image_path{image_name};
  const std::string outname =
      image_path.stem().string() + "_c2g" + image_path.extension().string();
  const std::string outname_color = image_path.stem().string() + "_c2g_color" +
                                    image_path.extension().string();

  // load the image
  ColorImage initial_image(theta, alpha, quantize);
  initial_image.load(sourceData, sourceWidth,  sourceHeight);
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
      const TimeBench bench{"Create a quantized image"};
      // TODO: Fix
      // const auto quantized = quantify_image(source, q_colors);
      // initial_image.load_quant_data(quantized);
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
