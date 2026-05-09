#include "color_image.hpp"

#include <cmath>
#include <iostream>
#include <map>

#include "cuda_kernels.hpp"
#include "utils.hpp"

ColorImage::ColorImage(float a_theta, float a_alpha, bool a_quantize)
    : mTheta{a_theta}, mAlpha{a_alpha}, mQuantize{a_quantize} {}

std::vector<float> ColorImage::calc_d() {
  if (mQuantize) {
    return cuda_calc_d_q(mN, mData.data(), mQdata, mTheta, mAlpha);
  }
  return cuda_calc_d_nq(mN, mData.data(), mTheta, mAlpha);
}

std::vector<float> ColorImage::r_calc_d(int r) {
  return cuda_r_calc_d(mW, mH, mData.data(), mTheta, mAlpha, r);
}

void ColorImage::load(const std::vector<uint8_t>& imgData, const int width,
                      const int height) {
  using sven::rgb;

  mW = width;
  mH = height;

  mN = mW * mH;
  std::cout << "image loaded, w: " << mW << ", y: " << mH << ".\n";

  mData.resize(mN);

  for (int i = 0; i < mN; i++) {
    mData[i] =
        amy_lab(rgb(imgData[3 * i], imgData[3 * i + 1], imgData[3 * i + 2]));
  }
}

void ColorImage::load_quant_data(const std::vector<uint8_t>& imgData,
                                 const int width, const int height) {
  using sven::rgb;

  mW = width;
  mH = height;

  std::vector<rgb> colors;

  mData.resize(mN);

  for (int i = 0; i < mN; i++) {
    colors.emplace_back(imgData[3 * i], imgData[3 * i + 1], imgData[3 * i + 2]);
    mData[i] =
        amy_lab(rgb(imgData[3 * i], imgData[3 * i + 1], imgData[3 * i + 2]));
  }

  mN = mW * mH;
  std::cout << "quantized image loaded, w: " << mW << ", y: " << mH << ".\n";

  mQdata.clear();

  using namespace std;

  map<rgb, int> q;
  map<rgb, int>::iterator r;
  for (int i = 0; i < mN; i++) {
    r = q.find(colors[i]);
    if (r == q.end())
      q[colors[i]] = 1;
    else
      r->second++;
  }

  std::cout << "quantized image appears to use " << q.size() << " colors.\n";
  mQdata.resize(q.size());
  int i = 0;
  for (i = 0, r = q.begin(); r != q.end(); ++r, i++) {
    mQdata[i] = amy_lab_int(amy_lab(r->first), r->second);
  }
}
