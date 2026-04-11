#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "color_image.hpp"
#include "gray_image.hpp"
#include "utils.hpp"

#ifdef HAS_OPENCL
#include "opencl_context.hpp"
#endif

// Generate a synthetic RGB image of given size
static std::vector<uint8_t> generateImage(int width, int height,
                                          unsigned seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<uint8_t> img(3 * width * height);
  for (auto& v : img) v = static_cast<uint8_t>(dist(gen));
  return img;
}

struct BenchResult {
  double load_us;
  double calc_d_us;
  double solve_us;
  double total_us;
};

static double median(std::vector<double>& v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2 == 0) ? (v[n / 2 - 1] + v[n / 2]) / 2.0 : v[n / 2];
}

using Clock = std::chrono::high_resolution_clock;

static double elapsed_us(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

// Run CPU benchmark for full solve
static BenchResult benchCPU_full(const std::vector<uint8_t>& img, int w, int h,
                                 float theta, float alpha) {
  BenchResult r{};

  auto t0 = Clock::now();
  ColorImage ci(theta, alpha, false);
  ci.load(img, w, h);
  auto t1 = Clock::now();
  r.load_us = elapsed_us(t0, t1);

  GrayImage gi(ci);

  t0 = Clock::now();
  auto d = ci.calc_d();
  t1 = Clock::now();
  r.calc_d_us = elapsed_us(t0, t1);

  t0 = Clock::now();
  gi.complete_solve(d);
  gi.post_solve(ci);
  t1 = Clock::now();
  r.solve_us = elapsed_us(t0, t1);

  r.total_us = r.load_us + r.calc_d_us + r.solve_us;
  return r;
}

// Run CPU benchmark for neighborhood solve
static BenchResult benchCPU_neigh(const std::vector<uint8_t>& img, int w, int h,
                                  float theta, float alpha, int mu) {
  BenchResult r{};

  auto t0 = Clock::now();
  ColorImage ci(theta, alpha, false);
  ci.load(img, w, h);
  auto t1 = Clock::now();
  r.load_us = elapsed_us(t0, t1);

  GrayImage gi(ci);

  t0 = Clock::now();
  auto d = ci.r_calc_d(mu);
  t1 = Clock::now();
  r.calc_d_us = elapsed_us(t0, t1);

  t0 = Clock::now();
  gi.r_solve(d, mu);
  gi.post_solve(ci);
  t1 = Clock::now();
  r.solve_us = elapsed_us(t0, t1);

  r.total_us = r.load_us + r.calc_d_us + r.solve_us;
  return r;
}

#ifdef HAS_OPENCL
// Run GPU benchmark for full solve
static BenchResult benchGPU_full(const std::vector<uint8_t>& img, int w, int h,
                                 float theta, float alpha, OpenCLContext& ocl) {
  BenchResult r{};

  auto t0 = Clock::now();
  ColorImage ci(theta, alpha, false);
  ci.load_ocl(img, w, h, ocl);
  auto t1 = Clock::now();
  r.load_us = elapsed_us(t0, t1);

  GrayImage gi(ci);

  t0 = Clock::now();
  auto d = ci.calc_d_ocl(ocl);
  t1 = Clock::now();
  r.calc_d_us = elapsed_us(t0, t1);

  t0 = Clock::now();
  gi.complete_solve(d);  // sequential — stays CPU
  gi.post_solve(ci);
  t1 = Clock::now();
  r.solve_us = elapsed_us(t0, t1);

  r.total_us = r.load_us + r.calc_d_us + r.solve_us;
  return r;
}

// Run GPU benchmark for neighborhood solve
static BenchResult benchGPU_neigh(const std::vector<uint8_t>& img, int w, int h,
                                  float theta, float alpha, int mu,
                                  OpenCLContext& ocl) {
  BenchResult r{};

  auto t0 = Clock::now();
  ColorImage ci(theta, alpha, false);
  ci.load_ocl(img, w, h, ocl);
  auto t1 = Clock::now();
  r.load_us = elapsed_us(t0, t1);

  GrayImage gi(ci);

  t0 = Clock::now();
  auto d = ci.r_calc_d_ocl(mu, ocl);
  t1 = Clock::now();
  r.calc_d_us = elapsed_us(t0, t1);

  t0 = Clock::now();
  gi.r_solve_ocl(d, mu, ocl);
  gi.post_solve(ci);
  t1 = Clock::now();
  r.solve_us = elapsed_us(t0, t1);

  r.total_us = r.load_us + r.calc_d_us + r.solve_us;
  return r;
}
#endif

static void printRow(const char* label, const BenchResult& r) {
  std::cout << std::setw(18) << label << " | " << std::setw(10)
            << std::fixed << std::setprecision(0) << r.load_us << " | "
            << std::setw(10) << r.calc_d_us << " | " << std::setw(10)
            << r.solve_us << " | " << std::setw(10) << r.total_us << "\n";
}

static void printHeader() {
  std::cout << std::setw(18) << "Path" << " | " << std::setw(10) << "Load(us)"
            << " | " << std::setw(10) << "CalcD(us)" << " | " << std::setw(10)
            << "Solve(us)" << " | " << std::setw(10) << "Total(us)"
            << "\n";
  std::cout << std::string(70, '-') << "\n";
}

int main() {
  constexpr float theta = 44.977226f * static_cast<float>(M_PI) / 180.0f;
  constexpr float alpha = 10.0f;
  constexpr int mu = 10;
  constexpr int runs = 3;

  struct ImageSize {
    int w, h;
    const char* name;
  };

  // Test with multiple image sizes to show scaling
  const ImageSize sizes[] = {
      {1, 172, "1x172 (test)"},
      {64, 64, "64x64 (4K px)"},
      {128, 128, "128x128 (16K px)"},
      {256, 256, "256x256 (65K px)"},
      {512, 512, "512x512 (262K px)"},
  };

#ifdef HAS_OPENCL
  OpenCLContext ocl;
  if (!ocl.isValid()) {
    std::cerr << "OpenCL initialization failed.\n";
    return 1;
  }
#endif

  // =========================================================
  // Full solve benchmark (O(N^2) — only for small images)
  // =========================================================
  std::cout << "\n=== FULL SOLVE (all-pairs, O(N^2)) ===\n\n";

  for (const auto& sz : sizes) {
    const int N = sz.w * sz.h;
    // Skip large images for full solve — would take forever on CPU
    if (N > 20000) {
      std::cout << sz.name << ": skipped (N=" << N
                << " too large for O(N^2) full solve)\n\n";
      continue;
    }

    const auto img = generateImage(sz.w, sz.h);

    std::cout << sz.name << " (N=" << N << ")\n";
    printHeader();

    // Warmup + median of runs
    std::vector<double> cpu_totals, gpu_totals;
    BenchResult cpu_best{}, gpu_best{};

    for (int i = 0; i < runs; i++) {
      auto r = benchCPU_full(img, sz.w, sz.h, theta, alpha);
      cpu_totals.push_back(r.total_us);
      if (i == 0 || r.total_us < cpu_best.total_us) cpu_best = r;
    }
    printRow("CPU (best)", cpu_best);

#ifdef HAS_OPENCL
    for (int i = 0; i < runs; i++) {
      auto r = benchGPU_full(img, sz.w, sz.h, theta, alpha, ocl);
      gpu_totals.push_back(r.total_us);
      if (i == 0 || r.total_us < gpu_best.total_us) gpu_best = r;
    }
    printRow("GPU (best)", gpu_best);

    double speedup = cpu_best.total_us / gpu_best.total_us;
    double calc_d_speedup = cpu_best.calc_d_us / gpu_best.calc_d_us;
    std::cout << "  => Total speedup: " << std::fixed << std::setprecision(2)
              << speedup << "x"
              << "  |  calc_d speedup: " << calc_d_speedup << "x\n";
#endif
    std::cout << "\n";
  }

  // =========================================================
  // Neighborhood solve benchmark (O(r^2 * N * iters))
  // =========================================================
  std::cout << "\n=== NEIGHBORHOOD SOLVE (mu=" << mu << ") ===\n\n";

  for (const auto& sz : sizes) {
    const int N = sz.w * sz.h;
    const auto img = generateImage(sz.w, sz.h);

    std::cout << sz.name << " (N=" << N << ")\n";
    printHeader();

    std::vector<double> cpu_totals, gpu_totals;
    BenchResult cpu_best{}, gpu_best{};

    for (int i = 0; i < runs; i++) {
      auto r = benchCPU_neigh(img, sz.w, sz.h, theta, alpha, mu);
      cpu_totals.push_back(r.total_us);
      if (i == 0 || r.total_us < cpu_best.total_us) cpu_best = r;
    }
    printRow("CPU (best)", cpu_best);

#ifdef HAS_OPENCL
    for (int i = 0; i < runs; i++) {
      auto r = benchGPU_neigh(img, sz.w, sz.h, theta, alpha, mu, ocl);
      gpu_totals.push_back(r.total_us);
      if (i == 0 || r.total_us < gpu_best.total_us) gpu_best = r;
    }
    printRow("GPU (best)", gpu_best);

    double speedup = cpu_best.total_us / gpu_best.total_us;
    double calc_d_speedup = cpu_best.calc_d_us / gpu_best.calc_d_us;
    double solve_speedup = cpu_best.solve_us / gpu_best.solve_us;
    std::cout << "  => Total speedup: " << std::fixed << std::setprecision(2)
              << speedup << "x"
              << "  |  r_calc_d speedup: " << calc_d_speedup << "x"
              << "  |  r_solve speedup: " << solve_speedup << "x\n";
#endif
    std::cout << "\n";
  }

  return 0;
}
