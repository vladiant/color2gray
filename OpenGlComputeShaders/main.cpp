#include <args.h>

#include <charconv>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "amy_lab.hpp"
#include "bitmap.hpp"
#include "color_image.hpp"
#include "gl_compute.hpp"
#include "gray_image.hpp"
#include "time_bench.hpp"
#include "utils.hpp"

constexpr auto doc = R"doc(
color2gray algorithm BMP demo (OpenGL compute)
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

  const int N = sourceWidth * sourceHeight;

  // Quantized path: fall back to CPU (existing code)
  if (quantize) {
    std::cout << "q = " << q_colors << '\n';
    ColorImage initial_image(theta, alpha, quantize);
    initial_image.load(sourceData, sourceWidth, sourceHeight);
    {
      const TimeBench bench{"Create a quantized image"};
      const auto quantized = quantify_image(sourceData, q_colors);
      initial_image.load_quant_data(quantized, sourceWidth, sourceHeight);
    }
    GrayImage dest(initial_image);
    std::vector<float> d;
    {
      const TimeBench bench{"calc_d"};
      d = initial_image.calc_d();
    }
    {
      const TimeBench bench{"complete_solve"};
      dest.complete_solve(d);
    }
    {
      const TimeBench bench{"post_solve"};
      dest.post_solve(initial_image);
    }
    dest.save(outname.c_str());
    dest.saveColor(outname_color.c_str(), initial_image);
    return EXIT_SUCCESS;
  }

  // Initialize OpenGL compute
  GLCompute gl;
  if (!gl.init()) {
    std::cout << "Failed to initialize OpenGL compute\n";
    return EXIT_FAILURE;
  }

  // Convert RGB to LAB on GPU
  std::vector<float> labData;
  {
    const TimeBench bench{"GPU: RGB to LAB"};
    labData = gl.rgbToLab(sourceData, N);
  }

  const auto start = std::chrono::high_resolution_clock::now();

  std::vector<float> grayData(N);

  if (r) {
    // Neighborhood case: calc_d on GPU, Gauss-Seidel on CPU
    std::vector<float> d;
    {
      const TimeBench bench{"GPU: r_calc_d"};
      d = gl.calcDR(labData, sourceWidth, sourceHeight, r, theta, alpha);
    }

    // Initialize gray from L channel
    for (int i = 0; i < N; i++) grayData[i] = labData[4 * i];

    // Gauss-Seidel relaxation (sequential, stays on CPU)
    {
      const TimeBench bench{"CPU: r_solve"};
      constexpr int iters = 30;
      for (int k = 0; k < iters; k++) {
        for (int x = 0; x < sourceWidth; x++) {
          for (int y = 0; y < sourceHeight; y++) {
            float sum = 0;
            int count = 0;
            for (int xx = x - r; xx <= x + r; xx++) {
              if (xx < 0 || xx >= sourceWidth) continue;
              for (int yy = y - r; yy <= y + r; yy++) {
                if (yy >= sourceHeight || yy < 0) continue;
                sum += grayData[xx + yy * sourceWidth];
                count++;
              }
            }
            grayData[x + y * sourceWidth] =
                (d[x + sourceWidth * y] + sum) / static_cast<float>(count);
          }
        }
      }
    }
  } else {
    // Full pairwise case: all on GPU
    std::vector<float> d;
    {
      const TimeBench bench{"GPU: calc_d"};
      d = gl.calcD(labData, N, theta, alpha);
    }

    {
      const TimeBench bench{"GPU: complete_solve"};
      grayData = gl.completeSolve(labData, d, N);
    }
  }

  // Post-solve on CPU
  {
    const TimeBench bench{"post_solve"};
    float error = 0;
    for (int i = 0; i < N; i++) error += grayData[i] - labData[4 * i];
    error /= N;
    for (int i = 0; i < N; i++) grayData[i] -= error;
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto process_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "c2g completed in " << process_time << " milliseconds\n";

  // Save grayscale output
  {
    std::vector<uint8_t> bmpData;
    bmpData.reserve(3 * N);
    for (int i = 0; i < N; i++) {
      const sven::rgb rval = amy_lab(grayData[i], 0, 0).to_rgb();
      bmpData.push_back(rval.r);
      bmpData.push_back(rval.g);
      bmpData.push_back(rval.b);
    }
    writeBMP(outname, sourceWidth, sourceHeight, bmpData);
  }

  // Save color-adapted output
  {
    std::vector<uint8_t> bmpData;
    bmpData.reserve(3 * N);
    for (int i = 0; i < N; i++) {
      const sven::rgb rval =
          amy_lab(grayData[i], labData[4 * i + 1], labData[4 * i + 2])
              .to_rgb();
      bmpData.push_back(rval.r);
      bmpData.push_back(rval.g);
      bmpData.push_back(rval.b);
    }
    writeBMP(outname_color, sourceWidth, sourceHeight, bmpData);
  }

  return EXIT_SUCCESS;
}
