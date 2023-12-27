#include "gray_image.hpp"

#include <iostream>

#include "bitmap.hpp"

GrayImage::GrayImage(const ColorImage &s)
    : mData(s.getN()), mW(s.getW()), mH(s.getH()), mN(s.getN()) {
  const auto &data = s.getData();
  for (int i = 0; i < mN; i++) mData[i] = data[i].l;
}

void GrayImage::r_solve(const std::vector<float> &d, int r) {
  constexpr int iters = 30;
  int k = 0, x = 0, y = 0;

  for (k = 0; k < iters; k++) {
    // std::cout << "iter " << k << "\n";

    // perform a Gauss-Seidel relaxation.
    for (x = 0; x < mW; x++)
      for (y = 0; y < mH; y++) {
        float sum = 0;
        int count = 0;
        int xx = 0, yy = 0;

        for (xx = x - r; xx <= x + r; xx++) {
          if (xx < 0 || xx >= mW) continue;
          for (yy = y - r; yy <= y + r; yy++) {
            if (yy >= mH || yy < 0) continue;
            sum += mData[xx + yy * mW];
            count++;
          }
        }

        mData[x + y * mW] = (d[x + mW * y] + sum) / (float)count;
      }
  }
}

void GrayImage::complete_solve(const std::vector<float> &d) {
  for (int i = 1; i < mN; i++) {
    mData[i] = d[i] - d[i - 1] + mN * mData[i - 1];
    mData[i] /= (float)mN;
  }
}

void GrayImage::post_solve(const ColorImage &s) {
  float error = 0;
  const auto &data = s.getData();
  for (int i = 0; i < mN; i++) error += mData[i] - data[i].l;
  error /= mN;
  for (int i = 0; i < mN; i++) mData[i] = mData[i] - error;
}

std::vector<uint8_t> GrayImage::save(const char *fname) const {
  std::vector<uint8_t> bmpData;
  bmpData.reserve(3 * mN);
  for (int i = 0; i < mN; i++) {
    const sven::rgb rval = amy_lab(mData[i], 0, 0).to_rgb();
    bmpData.push_back(rval.r);
    bmpData.push_back(rval.g);
    bmpData.push_back(rval.b);
  }

  if (fname) {
    writeBMP(fname, mW, mH, bmpData);
  }

  return bmpData;
}

std::vector<uint8_t> GrayImage::saveColor(const char *fname,
                                          const ColorImage &source) const {
  const auto &data = source.getData();

  std::vector<uint8_t> bmpData;
  bmpData.reserve(3 * mN);
  for (int i = 0; i < mN; i++) {
    const sven::rgb rval = amy_lab(mData[i], (data[i]).a, (data[i]).b).to_rgb();
    bmpData.push_back(rval.r);
    bmpData.push_back(rval.g);
    bmpData.push_back(rval.b);
  }

  if (fname) {
    writeBMP(fname, mW, mH, bmpData);
  }

  return bmpData;
}