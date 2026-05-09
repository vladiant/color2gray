#include "gray_image.hpp"

#include <iostream>

#include "bitmap.hpp"
#include "cuda_kernels.hpp"

GrayImage::GrayImage(const ColorImage &s)
    : mData(s.getN()), mW(s.getW()), mH(s.getH()), mN(s.getN()) {
  const auto &data = s.getData();
  for (int i = 0; i < mN; i++) mData[i] = data[i].l;
}

void GrayImage::r_solve(const std::vector<float> &d, int r) {
  cuda_r_solve(mW, mH, r, mData.data(), d.data());
}

void GrayImage::complete_solve(const std::vector<float> &d) {
  cuda_complete_solve(mN, mData.data(), d.data());
}

void GrayImage::post_solve(const ColorImage &s) {
  const auto &data = s.getData();
  std::vector<float> labL(mN);
  for (int i = 0; i < mN; ++i) labL[i] = data[i].l;
  cuda_post_solve(mN, mData.data(), labL.data());
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