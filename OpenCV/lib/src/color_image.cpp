#include "color_image.hpp"

#include <cmath>
#include <iostream>
#include <map>

#include "utils.hpp"

ColorImage::ColorImage(float a_theta, float a_alpha, bool a_quantize)
    : mTheta{a_theta}, mAlpha{a_alpha}, mQuantize{a_quantize} {}

std::vector<float> ColorImage::calc_d() {
  std::vector<float> d(mN);
  for (int i = 0; i < mN; i++) d[i] = 0;

  if (mQuantize) {
    for (int i = 0; i < mN; i++)
      for (size_t p = 0; p < mQdata.size(); p++) d[i] += calc_qdelta(i, p);

  } else {
    // more obvious but slower code for the unquantized full solve.
    // for (int i = 0; i < mN; i++)
    //   for (int j = 0; j < mN; j++) {
    //     float delta = calc_delta(i, j);
    //     d[i] += delta;
    //   }

    for (int i = 0; i < mN; i++)
      for (int j = i + 1; j < mN; j++) {
        float delta = calc_delta(i, j);
        d[i] += delta;
        d[j] -= delta;
      }
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
          float delta = calc_delta(i, j);
          d[i] += delta;
          d[j] -= delta;
        }
      }
    }
  return d;
}

float ColorImage::calc_delta(int i, int j) const {
  const amy_lab &a = data[i];
  const amy_lab &b = data[j];

  float dL = a.l - b.l;
  float dC = crunch(std::sqrt(sq(a.a - b.a) + sq(a.b - b.b)));

  if (fabsf(dL) > dC) return dL;
  return dC *
         ((std::cos(mTheta) * (a.a - b.a) + std::sin(mTheta) * (a.b - b.b)) > 0
              ? 1
              : -1);
}

float ColorImage::calc_qdelta(int i, int p) const {
  const amy_lab &a = data[i];
  const amy_lab &b = mQdata[p].first;

  float dL = a.l - b.l;
  float dC = crunch(std::sqrt(sq(a.a - b.a) + sq(a.b - b.b)));

  if (fabsf(dL) > dC) return mQdata[p].second * dL;
  return mQdata[p].second * dC *
         ((std::cos(mTheta) * (a.a - b.a) + std::sin(mTheta) * (a.b - b.b)) > 0
              ? 1
              : -1);
}

float ColorImage::crunch(float chrom_dist) const {
  return mAlpha == 0 ? 0 : mAlpha * std::tanh(chrom_dist / mAlpha);
}

void ColorImage::load(const cv::Mat3b &source) {
  using sven::rgb;

  mW = source.cols;
  mH = source.rows;

  mN = mW * mH;
  std::cout << "image loaded, w: " << mW << ", y: " << mH << ".\n";

  data.resize(mN);

  auto it = source.begin();
  for (int i = 0; i < mN; i++, ++it) {
    const cv::Vec3b color = *it;
    data[i] = amy_lab(rgb(color[2], color[1], color[0]));
  }
}

void ColorImage::load_quant_data(const cv::Mat3b &source) {
  using sven::rgb;

  mW = source.cols;
  mH = source.rows;

  std::vector<rgb> colors;

  data.resize(mN);

  auto it = source.begin();
  for (int i = 0; i < mN; i++, ++it) {
    const cv::Vec3b color = *it;
    colors.emplace_back(color[2], color[1], color[0]);
    data[i] = amy_lab(rgb(color[2], color[1], color[0]));
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
