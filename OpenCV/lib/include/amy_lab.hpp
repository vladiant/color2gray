#pragma once

#include "amy_xyz.hpp"
#include "rgb.hpp"

struct amy_lab {
  float l, a, b;

  amy_lab() = default;
  amy_lab(float l, float a, float b);
  explicit amy_lab(const sven::rgb &c);
  explicit amy_lab(const amy_xyz &c);

  sven::rgb to_rgb() const;
};