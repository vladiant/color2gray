#include "utils.hpp"

#include <iostream>
#include <opencv2/ml.hpp>

#include "amy_lab.hpp"

cv::Mat3b quantify_image(const cv::Mat3b& source, int q_colors) {
  cv::Mat points;
  source.convertTo(points, CV_32F);
  points = points.reshape(3, source.rows * source.cols);
  cv::TermCriteria condition(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                             20, 1.0);
  cv::Mat labels;
  cv::Mat centers;
  cv::kmeans(points, q_colors, labels, condition, 10, cv::KMEANS_RANDOM_CENTERS,
             centers);

  for (int i = 0; i < points.rows; i++) {
    points.at<cv::Point3f>(i, 0) =
        centers.at<cv::Point3f>(labels.at<int>(i, 0), 0);
  }
  points = points.reshape(3, source.rows);
  points.convertTo(points, CV_8U);

  return points;
}

// Computational functions

float crunch(float aChromDist, float aAlpha) {
  return aAlpha == 0 ? 0 : aAlpha * std::tanh(aChromDist / aAlpha);
}

float calc_qdelta(int i, int p, const amy_lab* aData,
                  const std::vector<std::pair<amy_lab, int>>& aQdata,
                  float aTheta, float aAlpha) {
  const amy_lab& a = aData[i];
  const amy_lab& b = aQdata[p].first;

  const float dL = a.l - b.l;
  const float dC = crunch(std::sqrt(sq(a.a - b.a) + sq(a.b - b.b)), aAlpha);

  if (fabsf(dL) > dC) {
    return aQdata[p].second * dL;
  }

  return aQdata[p].second * dC *
         ((std::cos(aTheta) * (a.a - b.a) + std::sin(aTheta) * (a.b - b.b)) > 0
              ? 1
              : -1);
}

float calc_delta(int i, int j, const amy_lab* aData, float aTheta,
                 float aAlpha) {
  const amy_lab& a = aData[i];
  const amy_lab& b = aData[j];

  float dL = a.l - b.l;
  float dC = crunch(std::sqrt(sq(a.a - b.a) + sq(a.b - b.b)), aAlpha);

  if (fabsf(dL) > dC) {
    return dL;
  }

  return dC *
         ((std::cos(aTheta) * (a.a - b.a) + std::sin(aTheta) * (a.b - b.b)) > 0
              ? 1
              : -1);
}

void calc_d_nq(int mN, const amy_lab* aData, float aTheta, float aAlpha,
               float* d) {
  for (int i = 0; i < mN; i++) d[i] = 0;

  // more obvious but slower code for the unquantized full solve.
  // for (int i = 0; i < mN; i++)
  //   for (int j = 0; j < mN; j++) {
  //     float delta = calc_delta(i, j, aData, aTheta, aAlpha);
  //     d[i] += delta;
  //   }

  for (int i = 0; i < mN; i++)
    for (int j = i + 1; j < mN; j++) {
      float delta = calc_delta(i, j, aData, aTheta, aAlpha);
      d[i] += delta;
      d[j] -= delta;
    }
}

void calc_d_q(int mN, const amy_lab* aData,
              const std::vector<std::pair<amy_lab, int>>& aQdata, float aTheta,
              float aAlpha, float* d) {
  for (int i = 0; i < mN; i++) d[i] = 0;

  for (int i = 0; i < mN; i++)
    for (size_t p = 0; p < aQdata.size(); p++)
      d[i] += calc_qdelta(i, p, aData, aQdata, aTheta, aAlpha);
}