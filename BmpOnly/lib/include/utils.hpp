#pragma once

#include <cstdint>
#include <vector>

#include "amy_lab.hpp"

struct amy_lab;

inline float clamp(float x, float x_min, float x_max) {
  if (x < x_min)
    return (x_min);
  else if (x > x_max)
    return (x_max);
  else
    return (x);
}

inline float sq(float s) { return s * s; }

std::vector<uint8_t> quantify_image(const std::vector<uint8_t>& source,
                                    int q_colors);

// Computational functions
float crunch(float aChromDist, float aAlpha);

float calc_qdelta(int i, int p, const amy_lab* aData,
                  const std::vector<std::pair<amy_lab, int>>& aQdata,
                  float aTheta, float aAlpha);

float calc_delta(int i, int j, const amy_lab* aData, float aTheta,
                 float aAlpha);

void calc_d_nq(int mN, const amy_lab* aData, float aTheta, float aAlpha,
               float* d);

void calc_d_q(int mN, const amy_lab* aData,
              const std::vector<std::pair<amy_lab, int>>& aQdata, float aTheta,
              float aAlpha, float* d);