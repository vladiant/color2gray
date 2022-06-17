#include "gray_image.hpp"

#include <opencv2/imgcodecs.hpp>

GrayImage::GrayImage(ColorImage &s) : data(s.N), w(s.w), h(s.h), N(s.N) {
  for (int i = 0; i < N; i++) data[i] = (s.data)[i].l;
}

void GrayImage::r_solve(const std::vector<float> &d, int r) {
  const int iters = 30;
  int k, x, y;

  for (k = 0; k < iters; k++) {
    // printf("iter %d\n", k);

    // perform a Gauss-Seidel relaxation.
    for (x = 0; x < w; x++)
      for (y = 0; y < h; y++) {
        float sum = 0;
        int count = 0;
        int xx, yy;

        for (xx = x - r; xx <= x + r; xx++) {
          if (xx < 0 || xx >= w) continue;
          for (yy = y - r; yy <= y + r; yy++) {
            if (yy >= h || yy < 0) continue;
            sum += data[xx + yy * w];
            count++;
          }
        }

        data[x + y * w] = (d[x + w * y] + sum) / (float)count;
      }
  }
}

void GrayImage::complete_solve(const std::vector<float> &d) {
  for (int i = 1; i < N; i++) {
    data[i] = d[i] - d[i - 1] + N * data[i - 1];
    data[i] /= (float)N;
  }
}

void GrayImage::post_solve(const ColorImage &s) {
  float error = 0;
  for (int i = 0; i < N; i++) error += data[i] - (s.data)[i].l;
  error /= N;
  for (int i = 0; i < N; i++) data[i] = data[i] - error;
}

cv::Mat GrayImage::save(const char *fname) const {
  cv::Mat3b out_image(h, w);
  auto it = out_image.begin();
  for (int i = 0; i < N; i++, ++it) {
    const sven::rgb rval = amy_lab(data[i], 0, 0).to_rgb();
    *it = cv::Vec3b(rval.r, rval.g, rval.b);
  }

  cv::imwrite(fname, out_image);
  return out_image;
}

cv::Mat GrayImage::saveColor(const char *fname,
                             const ColorImage &source) const {
  cv::Mat3b out_image(h, w);
  auto it = out_image.begin();
  for (int i = 0; i < N; i++, ++it) {
    const sven::rgb rval =
        amy_lab(data[i], ((source.data)[i]).a, ((source.data)[i]).b).to_rgb();
    *it = cv::Vec3b(rval.b, rval.g, rval.r);
  }

  cv::imwrite(fname, out_image);
  return out_image;
}
