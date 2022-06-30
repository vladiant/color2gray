#include "amy_lab.hpp"

#include <cmath>

amy_lab::amy_lab(float l, float a, float b) : l(l), a(a), b(b) {}

amy_lab::amy_lab(const sven::rgb &c) {
  amy_xyz temp(c);
  *this = amy_lab(temp);
}

amy_lab::amy_lab(const amy_xyz &c) {
  using sven::one_third;

  float X_third = std::pow(c.X, (float)one_third);
  float Y_third = std::pow(c.Y, (float)one_third);
  float Z_third = std::pow(c.Z, (float)one_third);

  if (c.Y > 0.008856)
    l = (116.0f * (Y_third)) - 16.0f;
  else
    l = 903.3f * c.Y;

  a = 500.0f * ((X_third) - (Y_third));
  b = 200.0f * ((Y_third) - (Z_third));
}

sven::rgb amy_lab::to_rgb() const {
  float P = (l + 16.0) / 116.0;

  // Define as constants
  float Xn = 0.9513f;
  float Yn = 1.000f;
  float Zn = 1.0886f;

  float X = Xn * std::pow(P + (a / 500), 3);
  float Y = Yn * std::pow(P, 3);
  float Z = Zn * std::pow(P - (b / 200), 3);

  return amy_xyz(X, Y, Z).to_rgb();
}