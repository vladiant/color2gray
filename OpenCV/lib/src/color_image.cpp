#include "color_image.hpp"

#include <cmath>
#include <map>

#include "utils.hpp"

ColorImage::ColorImage(float a_theta, float a_alpha, bool a_quantize)
    : theta{a_theta}, alpha{a_alpha}, quantize{a_quantize} {}

std::vector<float> ColorImage::calc_d() {
  std::vector<float> d(N);
  for (int i = 0; i < N; i++) d[i] = 0;

  if (quantize) {
    for (int i = 0; i < N; i++)
      for (size_t p = 0; p < qdata.size(); p++) d[i] += calc_qdelta(i, p);

  } else {
    // more obvious but slower code for the unquantized full solve.
    // for(int i=0; i<N;i++) for(j=0;j<N;j++) {
    //         float delta = calc_delta(i,j);
    //         d[i]+=delta;
    // }

    for (int i = 0; i < N; i++)
      for (int j = i + 1; j < N; j++) {
        float delta = calc_delta(i, j);
        d[i] += delta;
        d[j] -= delta;
      }
  }
  return d;
}

std::vector<float> ColorImage::r_calc_d(int r) {
  std::vector<float> d(N);
  int i = 0;
  for (i = 0; i < N; i++) d[i] = 0;

  int x = 0, y = 0;
  for (x = 0; x < w; x++)
    for (y = 0; y < h; y++) {
      int xx = 0, yy = 0;

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
  float dC = crunch(std::sqrt(sq(a.a - b.a) + sq(a.b - b.b)));

  if (fabsf(dL) > dC) return dL;
  return dC *
         ((std::cos(theta) * (a.a - b.a) + std::sin(theta) * (a.b - b.b)) > 0 ? 1 : -1);
}

float ColorImage::calc_qdelta(int i, int p) const {
  const amy_lab &a = data[i];
  const amy_lab &b = qdata[p].first;

  float dL = a.l - b.l;
  float dC = crunch(std::sqrt(sq(a.a - b.a) + sq(a.b - b.b)));

  if (fabsf(dL) > dC) return qdata[p].second * dL;
  return qdata[p].second * dC *
         ((std::cos(theta) * (a.a - b.a) + std::sin(theta) * (a.b - b.b)) > 0 ? 1 : -1);
}

float ColorImage::crunch(float chrom_dist) const {
  return alpha == 0 ? 0 : alpha * std::tanh(chrom_dist / alpha);
}

void ColorImage::load(const cv::Mat3b &source) {
  using sven::rgb;

  w = source.cols;
  h = source.rows;

  N = w * h;
  printf("image loaded, w: %d, y: %d.\n", w, h);

  data.resize(N);

  auto it = source.begin();
  for (int i = 0; i < N; i++, ++it) {
    const cv::Vec3b color = *it;
    data[i] = amy_lab(rgb(color[2], color[1], color[0]));
  }
}

void ColorImage::load_quant_data(const cv::Mat3b &source) {
  using sven::rgb;

  w = source.cols;
  h = source.rows;

  std::vector<rgb> colors;

  data.resize(N);

  auto it = source.begin();
  for (int i = 0; i < N; i++, ++it) {
    const cv::Vec3b color = *it;
    colors.emplace_back(color[2], color[1], color[0]);
    data[i] = amy_lab(rgb(color[2], color[1], color[0]));
  }

  N = w * h;
  printf("quantized image loaded, w: %d, y: %d.\n", w, h);

  qdata.clear();

  using namespace std;

  map<rgb, int> q;
  map<rgb, int>::iterator r;
  for (int i = 0; i < N; i++) {
    r = q.find(colors[i]);
    if (r == q.end())
      q[colors[i]] = 1;
    else
      r->second++;
  }

  printf("quantized image appears to use %zu colors.\n", q.size());
  qdata.resize(q.size());
  int i = 0;
  for (i = 0, r = q.begin(); r != q.end(); ++r, i++) {
    qdata[i] = amy_lab_int(amy_lab(r->first), r->second);
  }
}
