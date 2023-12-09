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

void ColorImage::load(const cv::Mat3b &source) {
  using sven::rgb;

  mW = source.cols;
  mH = source.rows;

  mN = mW * mH;
  std::cout << "image loaded, w: " << mW << ", y: " << mH << ".\n";

  mData.resize(mN);

  auto it = source.begin();
  for (int i = 0; i < mN; i++, ++it) {
    const cv::Vec3b color = *it;
    mData[i] = amy_lab(rgb(color[2], color[1], color[0]));
  }
}

void ColorImage::load_quant_data(const cv::Mat3b &source) {
  using sven::rgb;

  mW = source.cols;
  mH = source.rows;

  std::vector<rgb> colors;

  mData.resize(mN);

  auto it = source.begin();
  for (int i = 0; i < mN; i++, ++it) {
    const cv::Vec3b color = *it;
    colors.emplace_back(color[2], color[1], color[0]);
    mData[i] = amy_lab(rgb(color[2], color[1], color[0]));
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
