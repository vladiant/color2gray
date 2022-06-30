#include "amy_xyz.hpp"

#include "utils.hpp"

amy_xyz::amy_xyz(float X, float Y, float Z) : X(X), Y(Y), Z(Z) {}

amy_xyz::amy_xyz(const sven::rgb &c) {
  float B = c.b / 255.;
  float G = c.g / 255.;
  float R = c.r / 255.;

  X = 0.412453f * R + 0.357580f * G + 0.180423f * B;
  Y = 0.212671f * R + 0.715160f * G + 0.072169f * B;
  Z = 0.019334f * R + 0.119193f * G + 0.950227f * B;

  const float Xn = 0.9513f;
  const float Yn = 1.000f;
  const float Zn = 1.0886f;

  X = X / Xn;
  Y = Y / Yn;
  Z = Z / Zn;
}

sven::rgb amy_xyz::to_rgb() const {
  // for RGB [0,1] etc.
  float R = 3.240479f * X + -1.537150f * Y + -0.498535f * Z;
  float G = -0.969256f * X + 1.875992f * Y + 0.041556f * Z;
  float B = 0.055648f * X + -0.204043f * Y + 1.057311f * Z;
  R = clamp(R, 0.0, 1.0);
  G = clamp(G, 0.0, 1.0);
  B = clamp(B, 0.0, 1.0);

  return sven::rgb(static_cast<unsigned char>(R * 255),
                   static_cast<unsigned char>(G * 255),
                   static_cast<unsigned char>(B * 255));
}