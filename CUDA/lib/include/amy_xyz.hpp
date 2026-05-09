#pragma once

#include "rgb.hpp"

struct amy_xyz {
  float X, Y, Z;

  amy_xyz(float X, float Y, float Z);
  explicit amy_xyz(const sven::rgb &c);

  sven::rgb to_rgb() const;
};