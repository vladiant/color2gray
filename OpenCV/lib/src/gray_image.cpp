#include "gray_image.hpp"

#include <iostream>
#include <opencv2/imgcodecs.hpp>

GrayImage::GrayImage(ColorImage &s)
    : mData(s.mN), mW(s.mW), mH(s.mH), mN(s.mN) {
  for (int i = 0; i < mN; i++) mData[i] = (s.data)[i].l;
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
  for (int i = 0; i < mN; i++) error += mData[i] - (s.data)[i].l;
  error /= mN;
  for (int i = 0; i < mN; i++) mData[i] = mData[i] - error;
}

cv::Mat GrayImage::save(const char *fname) const {
  cv::Mat3b out_image(mH, mW);
  auto it = out_image.begin();
  for (int i = 0; i < mN; i++, ++it) {
    const sven::rgb rval = amy_lab(mData[i], 0, 0).to_rgb();
    *it = cv::Vec3b(rval.r, rval.g, rval.b);
  }

  if (fname) {
    cv::imwrite(fname, out_image);
  }

  return std::move(out_image);
}

cv::Mat GrayImage::saveColor(const char *fname,
                             const ColorImage &source) const {
  cv::Mat3b out_image(mH, mW);
  auto it = out_image.begin();
  for (int i = 0; i < mN; i++, ++it) {
    const sven::rgb rval =
        amy_lab(mData[i], ((source.data)[i]).a, ((source.data)[i]).b).to_rgb();
    *it = cv::Vec3b(rval.b, rval.g, rval.r);
  }

  if (fname) {
    cv::imwrite(fname, out_image);
  }

  return std::move(out_image);
}