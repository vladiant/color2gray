#pragma once

#include <chrono>
#include <string>

class TimeBench {
 public:
  TimeBench(std::string aLabel);

  ~TimeBench();

 private:
  const std::string mLabel;
  const std::chrono::high_resolution_clock::time_point mStartPoint =
      std::chrono::high_resolution_clock::now();
};