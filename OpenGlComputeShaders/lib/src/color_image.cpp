#include "color_image.hpp"

#include <cmath>
#include <iostream>
#include <map>

#include "utils.hpp"

ColorImage::ColorImage(float a_theta, float a_alpha, bool a_quantize)
    : mTheta{a_theta}, mAlpha{a_alpha}, mQuantize{a_quantize} {}

std::vector<float> ColorImage::calc_d() {
  std::vector<float> d(mN);
  if (mQuantize) {
    calc_d_q(mN, mData.data(), mQdata, mTheta, mAlpha, d.data());
  } else {
    calc_d_nq(mN, mData.data(), mTheta, mAlpha, d.data());
  }
  return d;
}

std::vector<float> ColorImage::r_calc_d(int r) {
  std::vector<float> d(mN);
  int i = 0;
  for (i = 0; i < mN; i++) d[i] = 0;

  int x = 0, y = 0;
  for (x = 0; x < mW; x++)
    for (y = 0; y < mH; y++) {
      int xx = 0, yy = 0;

      i = x + y * mW;

      for (xx = x - r; xx <= x + r; xx++) {
        if (xx < 0 || xx >= mW) continue;
        for (yy = y - r; yy <= y + r; yy++) {
          if (yy >= mH || yy < 0) continue;
          int j = xx + yy * mW;
          float delta = calc_delta(i, j, mData.data(), mTheta, mAlpha);
          d[i] += delta;
          d[j] -= delta;
        }
      }
    }
  return d;
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
