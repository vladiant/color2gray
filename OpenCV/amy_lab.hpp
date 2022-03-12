#pragma once

#include "amy_xyz.hpp"
#include "rgb.hpp"

struct amy_lab {
  float l, a, b;

  amy_lab() = default;
  amy_lab(float l, float a, float b);
  amy_lab(const sven::rgb &c);
  amy_lab(amy_xyz &c);

  sven::rgb to_rgb();
};