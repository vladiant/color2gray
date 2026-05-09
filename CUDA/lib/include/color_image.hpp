#pragma once

#include <cstdint>
#include <vector>

#include "amy_lab.hpp"

struct ColorImage {
  ColorImage(float a_theta, float a_alpha, bool a_quantize);

  void load_quant_data(const std::vector<uint8_t>& imgData, const int width,
                       const int height);

  std::vector<float> calc_d();

  std::vector<float> r_calc_d(int r);

  void load(const std::vector<uint8_t>& imgData, const int width,
            const int height);

  const std::vector<amy_lab>& getData() const { return mData; }

  int getW() const { return mW; }
  int getH() const { return mH; }
  int getN() const { return mN; }

 private:
  int mW{}, mH{}, mN{};
  std::vector<amy_lab> mData;

  typedef std::pair<amy_lab, int> amy_lab_int;
  std::vector<amy_lab_int> mQdata;

  int mColors{};

  float mTheta{};
  float mAlpha{};
  bool mQuantize{};
};
