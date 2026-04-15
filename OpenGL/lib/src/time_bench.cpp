#include "time_bench.hpp"

#include <iostream>

TimeBench::TimeBench(std::string aLabel) : mLabel{std::move(aLabel)} {}

TimeBench::~TimeBench() {
  const auto endPoint = std::chrono::high_resolution_clock::now();
  const auto processTime =
      std::chrono::duration_cast<std::chrono::microseconds>(endPoint -
                                                            mStartPoint)
          .count();
  std::cout << mLabel << " : " << processTime << " microseconds\n";
}