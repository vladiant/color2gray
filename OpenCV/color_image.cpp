#include "color_image.hpp"

#include <cmath>

#include "utils.hpp"

ColorImage::ColorImage(float a_theta, float a_alpha, bool a_quantize)
    : theta{a_theta}, alpha{a_alpha}, quantize{a_quantize} {}

float *ColorImage::calc_d() {
  float *d = new float[N];
  int i, j;
  for (i = 0; i < N; i++) d[i] = 0;

  if (quantize) {
    int p;
    for (i = 0; i < N; i++)
      for (p = 0; p < qdata.size(); p++) d[i] += calc_qdelta(i, p);

  } else {
    // more obvious but slower code for the unquantized full solve.
    /*for( i=0; i<N;i++) for(j=0;j<N;j++) {
            float delta = calc_delta(i,j);
            d[i]+=delta;
    }*/

    for (i = 0; i < N; i++)
      for (j = i + 1; j < N; j++) {
        float delta = calc_delta(i, j);
        d[i] += delta;
        d[j] -= delta;
      }
  }
  return d;
}

float *ColorImage::r_calc_d(int r) {
  float *d = new float[N];
  int i;
  for (i = 0; i < N; i++) d[i] = 0;

  int x, y;
  for (x = 0; x < w; x++)
    for (y = 0; y < h; y++) {
      int xx, yy;

      i = x + y * w;

      for (xx = x - r; xx <= x + r; xx++) {
        if (xx < 0 || xx >= w) continue;
        for (yy = y - r; yy <= y + r; yy++) {
          if (yy >= h || yy < 0) continue;
          int j = xx + yy * w;
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
  float dC = crunch(sqrt(sq(a.a - b.a) + sq(a.b - b.b)));

  if (fabsf(dL) > dC) return dL;
  return dC *
         ((cos(theta) * (a.a - b.a) + sin(theta) * (a.b - b.b)) > 0 ? 1 : -1);
}

float ColorImage::calc_qdelta(int i, int p) const {
  const amy_lab &a = data[i];
  const amy_lab &b = qdata[p].first;

  float dL = a.l - b.l;
  float dC = crunch(sqrt(sq(a.a - b.a) + sq(a.b - b.b)));

  if (fabsf(dL) > dC) return qdata[p].second * dL;
  return qdata[p].second * dC *
         ((cos(theta) * (a.a - b.a) + sin(theta) * (a.b - b.b)) > 0 ? 1 : -1);
}

float ColorImage::crunch(float chrom_dist) const {
  return alpha == 0 ? 0 : alpha * tanh(chrom_dist / alpha);
}

void ColorImage::load(const cv::Mat3b &source) {
  using sven::rgb;

  if (data) delete[] data;

  w = source.cols;
  h = source.rows;

  N = w * h;
  printf("image loaded, w: %d, y: %d.\n", w, h);

  data = new amy_lab[N];

  auto it = source.begin();
  for (int i = 0; i < N; i++, ++it) {
    const cv::Vec3b color = *it;
    data[i] = amy_lab(rgb(color[2], color[1], color[0]));
  }
}