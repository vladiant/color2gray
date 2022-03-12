#include <cmath>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "color_image.hpp"
#include "gray_image.hpp"

constexpr auto doc = R"doc(
color2gray algorithm OpenCV demo
)doc";

constexpr float d2r = M_PI / 180.0;
constexpr float r2d = 180.0 / M_PI;
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

  cv::Mat source = cv::imread(image_name, cv::IMREAD_COLOR);
  if (source.empty()) {
    std::cout << "Failed to load file: " << image_name << '\n';
    return EXIT_FAILURE;
  }

  // Read parameters
  const float theta_deg = parser.get<float>("theta");
  std::cout << "Theta = " << theta_deg << '\n';
  const float theta = theta_deg * d2r;

  const float alpha = parser.get<float>("alpha");
  std::cout << "Alpha = " << alpha << '\n';

  const int r = parser.get<int>("alpha");
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
  ColorImage source1(theta, alpha, quantize);
  source1.load(source);
  GrayImage dest(source1);

  // solve, either using the complete case or the neighboorhod case.
  float* d;
  if (r) {
    d = source1.r_calc_d(r);
    dest.r_solve(d, r);
  } else {
    if (quantize) {
      char magick_string[256];
      // sprintf(magick_string,"convert -colors %d %s
      // quant-%s",q_colors,fname,fname);
      printf("\nUsing image magick to create a quantized image: \n%s\n",
             magick_string);
      system(magick_string);
      printf("done.\n\n");
      // sprintf(magick_string,"quant-%s",fname);
      // source1.load_quant_data(magick_string);
    }

    d = source1.calc_d();
    dest.complete_solve(d);
  }
  dest.post_solve(source1);

  auto gray_image = dest.save(outname.c_str());
  auto color_image = dest.saveColor(outname_color.c_str(), source1);

  cv::namedWindow("input", cv::WINDOW_NORMAL);
  cv::imshow("input", source);

  cv::namedWindow("gray", cv::WINDOW_NORMAL);
  cv::imshow("gray", gray_image);

  cv::namedWindow("color", cv::WINDOW_NORMAL);
  cv::imshow("color", color_image);

  while (true) {
    char ch = cv::waitKey(0);
    if (ch == 27) {
      break;
    }
  }

  delete[] d;

  return EXIT_SUCCESS;
}
