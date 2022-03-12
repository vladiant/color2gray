#pragma once

namespace sven {
struct rgb {
  unsigned char r, g, b;
  rgb() {}
  rgb(unsigned char r, unsigned char g, unsigned char b) : r(r), g(g), b(b) {}
  bool operator<(const rgb &c) const {
    if (r < c.r) return true;
    if (r > c.r) return false;
    if (g < c.g) return true;
    if (g > c.g) return false;
    return b < c.b;
  }
};

const float one_third =
    0.33333333333333333333333333333333333333333333333333333333333333;
}  // namespace sven
