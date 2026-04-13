#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdint>
#include <vector>

class GLCompute {
 public:
  GLCompute();
  ~GLCompute();

  GLCompute(const GLCompute&) = delete;
  GLCompute& operator=(const GLCompute&) = delete;

  bool init();

  // RGB (uint8, 3 per pixel) -> LAB (float, 4 per pixel: L,a,b,0)
  std::vector<float> rgbToLab(const std::vector<uint8_t>& rgbData, int n);

  // Compute d[] from LAB data (full pairwise, non-quantized)
  std::vector<float> calcD(const std::vector<float>& labData, int n,
                           float theta, float alpha);

  // Compute d[] from LAB data (neighborhood)
  std::vector<float> calcDR(const std::vector<float>& labData, int w, int h,
                            int r, float theta, float alpha);

  // Complete solve: g[i] = g0 + (d[i] - d0) / N
  std::vector<float> completeSolve(const std::vector<float>& labData,
                                   const std::vector<float>& d, int n);

 private:
  GLFWwindow* mWindow = nullptr;
  GLuint mRgbToLabProg = 0;
  GLuint mCalcDProg = 0;
  GLuint mCalcDRProg = 0;
  GLuint mCompleteSolveProg = 0;

  GLuint compileProgram(const char* source);
  void checkGLError(const char* context);
};
