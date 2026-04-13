#include "gl_compute.hpp"

#include <algorithm>
#include <iostream>
#include <string>

// ========== Compute Shader Sources ==========

static const char* kRgbToLabSource = R"glsl(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer RGBInput {
    float rgb_data[];
};

layout(std430, binding = 1) writeonly buffer LabOutput {
    vec4 lab_data[];
};

uniform int u_N;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(u_N)) return;

    float R = rgb_data[idx * 3 + 0];
    float G = rgb_data[idx * 3 + 1];
    float B = rgb_data[idx * 3 + 2];

    // RGB [0,1] to XYZ (normalized by white point)
    float X = (0.412453 * R + 0.357580 * G + 0.180423 * B) / 0.9513;
    float Y =  0.212671 * R + 0.715160 * G + 0.072169 * B;
    float Z = (0.019334 * R + 0.119193 * G + 0.950227 * B) / 1.0886;

    // XYZ to LAB
    float one_third = 1.0 / 3.0;
    float X_third = pow(X, one_third);
    float Y_third = pow(Y, one_third);
    float Z_third = pow(Z, one_third);

    float L;
    if (Y > 0.008856)
        L = 116.0 * Y_third - 16.0;
    else
        L = 903.3 * Y;

    float a = 500.0 * (X_third - Y_third);
    float b = 200.0 * (Y_third - Z_third);

    lab_data[idx] = vec4(L, a, b, 0.0);
}
)glsl";

static const char* kCalcDSource = R"glsl(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer LabInput {
    vec4 lab_data[];
};

layout(std430, binding = 1) writeonly buffer DOutput {
    float d_data[];
};

uniform int u_N;
uniform float u_theta;
uniform float u_alpha;
uniform int u_i_start;
uniform int u_i_end;

float crunch(float chromDist, float a) {
    if (a == 0.0) return 0.0;
    return a * tanh(chromDist / a);
}

void main() {
    int i = int(gl_GlobalInvocationID.x) + u_i_start;
    if (i >= u_i_end) return;

    vec4 ai = lab_data[i];
    float sum = 0.0;
    float cos_theta = cos(u_theta);
    float sin_theta = sin(u_theta);

    for (int j = 0; j < u_N; j++) {
        vec4 bj = lab_data[j];
        float dL = ai.x - bj.x;
        float da = ai.y - bj.y;
        float db = ai.z - bj.z;
        float chromDist = sqrt(da * da + db * db);
        float dC = crunch(chromDist, u_alpha);

        if (abs(dL) > dC) {
            sum += dL;
        } else {
            sum += dC * ((cos_theta * da + sin_theta * db) > 0.0 ? 1.0 : -1.0);
        }
    }

    d_data[i] = sum;
}
)glsl";

static const char* kCalcDRSource = R"glsl(
#version 430
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer LabInput {
    vec4 lab_data[];
};

layout(std430, binding = 1) writeonly buffer DOutput {
    float d_data[];
};

uniform int u_W;
uniform int u_H;
uniform int u_R;
uniform float u_theta;
uniform float u_alpha;

float crunch(float chromDist, float a) {
    if (a == 0.0) return 0.0;
    return a * tanh(chromDist / a);
}

void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= u_W || y >= u_H) return;

    int i = x + y * u_W;
    vec4 ai = lab_data[i];
    float sum = 0.0;
    float cos_theta = cos(u_theta);
    float sin_theta = sin(u_theta);

    for (int xx = x - u_R; xx <= x + u_R; xx++) {
        if (xx < 0 || xx >= u_W) continue;
        for (int yy = y - u_R; yy <= y + u_R; yy++) {
            if (yy < 0 || yy >= u_H) continue;
            int j = xx + yy * u_W;

            vec4 bj = lab_data[j];
            float dL = ai.x - bj.x;
            float da = ai.y - bj.y;
            float db = ai.z - bj.z;
            float chromDist = sqrt(da * da + db * db);
            float dC = crunch(chromDist, u_alpha);

            float delta;
            if (abs(dL) > dC) {
                delta = dL;
            } else {
                delta = dC * ((cos_theta * da + sin_theta * db) > 0.0 ? 1.0 : -1.0);
            }
            sum += delta;
        }
    }

    // Factor of 2 to match CPU antisymmetric double-update accumulation
    d_data[i] = 2.0 * sum;
}
)glsl";

static const char* kCompleteSolveSource = R"glsl(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer DInput {
    float d_data[];
};

layout(std430, binding = 1) writeonly buffer GOutput {
    float g_data[];
};

uniform int u_N;
uniform float u_g0;
uniform float u_d0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(u_N)) return;

    g_data[idx] = u_g0 + (d_data[idx] - u_d0) / float(u_N);
}
)glsl";

// ========== GLCompute Implementation ==========

GLCompute::GLCompute() = default;

GLCompute::~GLCompute() {
  if (mRgbToLabProg) glDeleteProgram(mRgbToLabProg);
  if (mCalcDProg) glDeleteProgram(mCalcDProg);
  if (mCalcDRProg) glDeleteProgram(mCalcDRProg);
  if (mCompleteSolveProg) glDeleteProgram(mCompleteSolveProg);
  if (mWindow) {
    glfwDestroyWindow(mWindow);
    glfwTerminate();
  }
}

bool GLCompute::init() {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW\n";
    return false;
  }

  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  mWindow = glfwCreateWindow(1, 1, "", nullptr, nullptr);
  if (!mWindow) {
    std::cerr << "Failed to create GLFW window (OpenGL 4.3 required)\n";
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(mWindow);

  glewExperimental = GL_TRUE;
  GLenum glewErr = glewInit();
  if (glewErr != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW: "
              << glewGetErrorString(glewErr) << "\n";
    return false;
  }

  // Clear any spurious error from glewExperimental
  glGetError();

  std::cout << "OpenGL " << glGetString(GL_VERSION) << "\n";
  std::cout << "GPU: " << glGetString(GL_RENDERER) << "\n";

  mRgbToLabProg = compileProgram(kRgbToLabSource);
  mCalcDProg = compileProgram(kCalcDSource);
  mCalcDRProg = compileProgram(kCalcDRSource);
  mCompleteSolveProg = compileProgram(kCompleteSolveSource);

  if (!mRgbToLabProg || !mCalcDProg || !mCalcDRProg || !mCompleteSolveProg) {
    std::cerr << "Failed to compile one or more compute shaders\n";
    return false;
  }

  return true;
}

GLuint GLCompute::compileProgram(const char* source) {
  GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  GLint success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    GLint logLen;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
    std::string log(logLen, '\0');
    glGetShaderInfoLog(shader, logLen, nullptr, log.data());
    std::cerr << "Shader compilation failed:\n" << log << "\n";
    glDeleteShader(shader);
    return 0;
  }

  GLuint program = glCreateProgram();
  glAttachShader(program, shader);
  glLinkProgram(program);

  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    GLint logLen;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLen);
    std::string log(logLen, '\0');
    glGetProgramInfoLog(program, logLen, nullptr, log.data());
    std::cerr << "Program linking failed:\n" << log << "\n";
    glDeleteProgram(program);
    glDeleteShader(shader);
    return 0;
  }

  glDeleteShader(shader);
  return program;
}

void GLCompute::checkGLError(const char* context) {
  GLenum err;
  while ((err = glGetError()) != GL_NO_ERROR) {
    std::cerr << "GL error at " << context << ": 0x" << std::hex << err
              << std::dec << "\n";
  }
}

std::vector<float> GLCompute::rgbToLab(const std::vector<uint8_t>& rgb,
                                       int n) {
  // Convert uint8 RGB to float RGB [0,1]
  std::vector<float> rgbFloat(n * 3);
  for (int i = 0; i < n * 3; i++) {
    rgbFloat[i] = rgb[i] / 255.0f;
  }

  GLuint buffers[2];
  glGenBuffers(2, buffers);

  // Input: float RGB
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               static_cast<GLsizeiptr>(rgbFloat.size() * sizeof(float)),
               rgbFloat.data(), GL_STATIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[0]);

  // Output: vec4 LAB
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               static_cast<GLsizeiptr>(n * 4 * sizeof(float)), nullptr,
               GL_STATIC_READ);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers[1]);

  glUseProgram(mRgbToLabProg);
  glUniform1i(glGetUniformLocation(mRgbToLabProg, "u_N"), n);

  GLuint groups = (n + 255) / 256;
  glDispatchCompute(groups, 1, 1);
  glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

  // Read back
  std::vector<float> labData(n * 4);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
  glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                     static_cast<GLsizeiptr>(labData.size() * sizeof(float)),
                     labData.data());

  glDeleteBuffers(2, buffers);
  checkGLError("rgbToLab");
  return labData;
}

std::vector<float> GLCompute::calcD(const std::vector<float>& lab, int n,
                                    float theta, float alpha) {
  GLuint buffers[2];
  glGenBuffers(2, buffers);

  // Input: LAB data (vec4 per pixel)
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               static_cast<GLsizeiptr>(lab.size() * sizeof(float)), lab.data(),
               GL_STATIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[0]);

  // Output: d[]
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               static_cast<GLsizeiptr>(n * sizeof(float)), nullptr,
               GL_DYNAMIC_COPY);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers[1]);

  glUseProgram(mCalcDProg);

  GLint locN = glGetUniformLocation(mCalcDProg, "u_N");
  GLint locTheta = glGetUniformLocation(mCalcDProg, "u_theta");
  GLint locAlpha = glGetUniformLocation(mCalcDProg, "u_alpha");
  GLint locIStart = glGetUniformLocation(mCalcDProg, "u_i_start");
  GLint locIEnd = glGetUniformLocation(mCalcDProg, "u_i_end");

  glUniform1i(locN, n);
  glUniform1f(locTheta, theta);
  glUniform1f(locAlpha, alpha);
  checkGLError("calcD:uniforms");

  // Tile over i only. Each dispatch processes the full j-range [0, N).
  // glFinish() after each dispatch to prevent GPU watchdog (TDR) reset
  // on display GPUs. Without this, accumulated back-to-back dispatches
  // exceed the NVIDIA TDR threshold and silently zero the output.
  constexpr int I_TILE = 10000;
  for (int i_start = 0; i_start < n; i_start += I_TILE) {
    int i_end = std::min(i_start + I_TILE, n);
    int i_count = i_end - i_start;
    GLuint groups = (i_count + 255) / 256;

    glUniform1i(locIStart, i_start);
    glUniform1i(locIEnd, i_end);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glFinish();
  }
  checkGLError("calcD:finish");

  // Read back
  std::vector<float> d(n);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
  glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                     static_cast<GLsizeiptr>(d.size() * sizeof(float)),
                     d.data());

  glDeleteBuffers(2, buffers);
  checkGLError("calcD");
  return d;
}

std::vector<float> GLCompute::calcDR(const std::vector<float>& lab, int w,
                                     int h, int r, float theta, float alpha) {
  int n = w * h;

  GLuint buffers[2];
  glGenBuffers(2, buffers);

  // Input: LAB data
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               static_cast<GLsizeiptr>(lab.size() * sizeof(float)), lab.data(),
               GL_STATIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[0]);

  // Output: d[]
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               static_cast<GLsizeiptr>(n * sizeof(float)), nullptr,
               GL_STATIC_READ);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers[1]);

  glUseProgram(mCalcDRProg);
  glUniform1i(glGetUniformLocation(mCalcDRProg, "u_W"), w);
  glUniform1i(glGetUniformLocation(mCalcDRProg, "u_H"), h);
  glUniform1i(glGetUniformLocation(mCalcDRProg, "u_R"), r);
  glUniform1f(glGetUniformLocation(mCalcDRProg, "u_theta"), theta);
  glUniform1f(glGetUniformLocation(mCalcDRProg, "u_alpha"), alpha);

  GLuint gx = (w + 15) / 16;
  GLuint gy = (h + 15) / 16;
  glDispatchCompute(gx, gy, 1);
  glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

  std::vector<float> d(n);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
  glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                     static_cast<GLsizeiptr>(d.size() * sizeof(float)),
                     d.data());

  glDeleteBuffers(2, buffers);
  checkGLError("calcDR");
  return d;
}

std::vector<float> GLCompute::completeSolve(const std::vector<float>& labData,
                                            const std::vector<float>& d,
                                            int n) {
  // Closed-form: g[i] = L[0] + (d[i] - d[0]) / N
  float g0 = labData[0];  // L channel of first pixel
  float d0 = d[0];

  GLuint buffers[2];
  glGenBuffers(2, buffers);

  // Input: d[]
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               static_cast<GLsizeiptr>(d.size() * sizeof(float)), d.data(),
               GL_STATIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[0]);

  // Output: g[]
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               static_cast<GLsizeiptr>(n * sizeof(float)), nullptr,
               GL_STATIC_READ);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers[1]);

  glUseProgram(mCompleteSolveProg);
  glUniform1i(glGetUniformLocation(mCompleteSolveProg, "u_N"), n);
  glUniform1f(glGetUniformLocation(mCompleteSolveProg, "u_g0"), g0);
  glUniform1f(glGetUniformLocation(mCompleteSolveProg, "u_d0"), d0);

  GLuint groups = (n + 255) / 256;
  glDispatchCompute(groups, 1, 1);
  glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

  std::vector<float> g(n);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
  glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                     static_cast<GLsizeiptr>(g.size() * sizeof(float)),
                     g.data());

  glDeleteBuffers(2, buffers);
  checkGLError("completeSolve");
  return g;
}
