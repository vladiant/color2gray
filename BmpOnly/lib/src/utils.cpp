#include "utils.hpp"

#include <iostream>
#include <cmath>
#include <random>

#include "amy_lab.hpp"

namespace {

  /// Random device seed
  constexpr auto kSeed = 1234;

  /// kMeans max iterations
  constexpr auto kMaxIterations = 20;

  /// kMeans threashold
  constexpr auto kEps = 1.0;

  struct ColorPoint {
    float R{0.0};
    float G{0.0};
    float B{0.0};
  };

} // namespace

std::vector<uint8_t> quantify_image(const std::vector<uint8_t>& source, int q_colors) {
  std::vector<ColorPoint> clusterCenters(q_colors);

  // Random generator of color cluster centers
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(kSeed);

  std::uniform_real_distribution<float> distR{0, 255};
  std::uniform_real_distribution<float> distG{0, 255};
  std::uniform_real_distribution<float> distB{0, 255};

  for (int i = 0; i < q_colors; i++) {
    clusterCenters[i].R = distR(gen);
    clusterCenters[i].G = distG(gen);
    clusterCenters[i].B = distB(gen);
  }

  const auto distance = [](const ColorPoint& lhs, const ColorPoint& rhs){
    return (lhs.R - rhs.R)*(lhs.R - rhs.R) + (lhs.G - rhs.G)*(lhs.G - rhs.G) + (lhs.B - rhs.B)*(lhs.B - rhs.B);
  };

  const auto getClosestCenterIndex = [&clusterCenters, &distance](const ColorPoint& point){
    int pos = -1;
    float minDistance = INFINITY;
    for (size_t i = 0; i < clusterCenters.size(); i++) {
      const auto currentDistance = distance(clusterCenters[i], point);
      if (currentDistance < minDistance) {
        minDistance = currentDistance;
        pos = i;
      }
    }

    return pos;
  };

  std::vector<int> labels(source.size()/3);
  for(int i = 0; i < kMaxIterations; i++) {
    // Set label for each point
    for(size_t i = 0; i < labels.size(); i++) {
      const ColorPoint currentPoint{.R = static_cast<float>(source[3*i]), .G=static_cast<float>(source[3*i + 1]), .B=static_cast<float>(source[3*i + 2])};
      labels[i] = getClosestCenterIndex(currentPoint);
    }

    // Recalculate centers
    std::vector<ColorPoint> newClusterCenters(q_colors);
    std::vector<size_t> labelsCount(q_colors);
    for(size_t j =0; j < labels.size(); j++) {
      const auto label = labels[j];
      labelsCount[label] += 1;
      const auto R = source[3*j];
      const auto G = source[3*j + 1];
      const auto B = source[3*j + 2];
      const auto currentLabelCount = labelsCount[label];
      newClusterCenters[label].R = (newClusterCenters[label].R*(currentLabelCount - 1) + R) / currentLabelCount;
      newClusterCenters[label].G = (newClusterCenters[label].G*(currentLabelCount - 1) + G) / currentLabelCount;
      newClusterCenters[label].B = (newClusterCenters[label].B*(currentLabelCount - 1) + B) / currentLabelCount;
    }

    // Check distance from old center
    float maxClusterCenterDistance = -INFINITY;
    for (size_t j = 0; j < clusterCenters.size(); j++) {
      const auto currentDistance = distance(clusterCenters[j], newClusterCenters[j]);
      if (currentDistance > maxClusterCenterDistance) {
        maxClusterCenterDistance = currentDistance;
      }
    }

    clusterCenters.swap(newClusterCenters);

    if (maxClusterCenterDistance < kEps) {
      break;
    }
  }

  std::vector<uint8_t> points(source.size());
  // Set color points
  for (size_t j = 0; j < source.size();) {
    const auto currentPoint = clusterCenters[labels[j/3]];
    points[j++] = currentPoint.R;
    points[j++] = currentPoint.G;
    points[j++] = currentPoint.B;
  }

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