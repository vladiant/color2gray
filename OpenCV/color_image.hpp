#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>

#include "amy_lab.hpp"

struct ColorImage {
  ColorImage(float a_theta, float a_alpha, bool a_quantize);

  std::vector<amy_lab> data;

  typedef std::pair<amy_lab, int> amy_lab_int;
  std::vector<amy_lab_int> qdata;

  int colors{};

  int w{}, h{}, N{};

  float calc_delta(int i, int j) const;
  float calc_qdelta(int i, int p) const;

  void load_quant_data(const cv::Mat3b &source);

  std::vector<float> calc_d();

  std::vector<float> r_calc_d(int r);

  void load(const cv::Mat3b &source);

 private:
  float theta{};
  float alpha{};
  bool quantize{};

  float crunch(float chrom_dist) const;
};
