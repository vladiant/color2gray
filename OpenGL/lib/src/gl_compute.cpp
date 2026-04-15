#include "gl_compute.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

GLFWwindow* gWindow = nullptr;
bool gInitialized = false;

GLuint gVAO = 0;
GLuint gVBO = 0;

GLuint gCalcDProgram = 0;
GLuint gCalcDQProgram = 0;
GLuint gRCalcDProgram = 0;
GLuint gRSolveProgram = 0;

constexpr int TILE_SIZE = 1024;
constexpr int FLUSH_INTERVAL = 16;

const float quadVerts[] = {-1.0f, -1.0f, 1.0f, -1.0f,
                           -1.0f, 1.0f,  1.0f, 1.0f};

// Shared vertex shader
const char* vsSrc = R"(
#version 130
in vec2 aPos;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

// Fragment shader: non-quantized calc_d (tiled, ping-pong accumulation)
const char* calcDNQ_fsSrc = R"(
#version 130
uniform sampler2D labTex;
uniform sampler2D prevAccum;
uniform int imageWidth;
uniform int imageHeight;
uniform int tileStart;
uniform int tileEnd;
uniform float theta;
uniform float alpha;
out vec4 fragColor;

float crunch_val(float chromDist, float a) {
    if (a == 0.0) return 0.0;
    return a * tanh(chromDist / a);
}

void main() {
    int ix = int(gl_FragCoord.x);
    int iy = int(gl_FragCoord.y);

    vec4 lab_i = texelFetch(labTex, ivec2(ix, iy), 0);
    float prev = texelFetch(prevAccum, ivec2(ix, iy), 0).r;

    float cosT = cos(theta);
    float sinT = sin(theta);
    float sum = 0.0;

    for (int j = tileStart; j < tileEnd; j++) {
        int jx = j % imageWidth;
        int jy = j / imageWidth;

        vec4 lab_j = texelFetch(labTex, ivec2(jx, jy), 0);

        float dL = lab_i.r - lab_j.r;
        float da = lab_i.g - lab_j.g;
        float db = lab_i.b - lab_j.b;
        float dC = crunch_val(sqrt(da * da + db * db), alpha);

        float delta;
        if (abs(dL) > dC) {
            delta = dL;
        } else {
            delta = dC * ((cosT * da + sinT * db) > 0.0 ? 1.0 : -1.0);
        }
        sum += delta;
    }

    fragColor = vec4(prev + sum, 0.0, 0.0, 1.0);
}
)";

// Fragment shader: quantized calc_d (single pass, loops over quantized colors)
const char* calcDQ_fsSrc = R"(
#version 130
uniform sampler2D labTex;
uniform sampler2D qDataTex;
uniform int imageWidth;
uniform int imageHeight;
uniform int numQColors;
uniform float theta;
uniform float alpha;
out vec4 fragColor;

float crunch_val(float chromDist, float a) {
    if (a == 0.0) return 0.0;
    return a * tanh(chromDist / a);
}

void main() {
    int ix = int(gl_FragCoord.x);
    int iy = int(gl_FragCoord.y);

    vec4 lab_i = texelFetch(labTex, ivec2(ix, iy), 0);

    float cosT = cos(theta);
    float sinT = sin(theta);
    float sum = 0.0;

    for (int p = 0; p < numQColors; p++) {
        vec4 qEntry = texelFetch(qDataTex, ivec2(p, 0), 0);
        float qL = qEntry.r;
        float qA = qEntry.g;
        float qB = qEntry.b;
        float qCount = qEntry.a;

        float dL = lab_i.r - qL;
        float da = lab_i.g - qA;
        float db = lab_i.b - qB;
        float dC = crunch_val(sqrt(da * da + db * db), alpha);

        float delta;
        if (abs(dL) > dC) {
            delta = qCount * dL;
        } else {
            delta = qCount * dC *
                    ((cosT * da + sinT * db) > 0.0 ? 1.0 : -1.0);
        }
        sum += delta;
    }

    fragColor = vec4(sum, 0.0, 0.0, 1.0);
}
)";

// Fragment shader: neighborhood calc_d with radius r
const char* rCalcD_fsSrc = R"(
#version 130
uniform sampler2D labTex;
uniform int imageWidth;
uniform int imageHeight;
uniform int radius;
uniform float theta;
uniform float alpha;
out vec4 fragColor;

float crunch_val(float chromDist, float a) {
    if (a == 0.0) return 0.0;
    return a * tanh(chromDist / a);
}

void main() {
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);

    vec4 lab_i = texelFetch(labTex, ivec2(x, y), 0);

    float cosT = cos(theta);
    float sinT = sin(theta);
    float sum = 0.0;

    int xStart = max(0, x - radius);
    int xEnd = min(imageWidth - 1, x + radius);
    int yStart = max(0, y - radius);
    int yEnd = min(imageHeight - 1, y + radius);

    for (int xx = xStart; xx <= xEnd; xx++) {
        for (int yy = yStart; yy <= yEnd; yy++) {
            vec4 lab_j = texelFetch(labTex, ivec2(xx, yy), 0);

            float dL = lab_i.r - lab_j.r;
            float da = lab_i.g - lab_j.g;
            float db = lab_i.b - lab_j.b;
            float dC = crunch_val(sqrt(da * da + db * db), alpha);

            float delta;
            if (abs(dL) > dC) {
                delta = dL;
            } else {
                delta = dC * ((cosT * da + sinT * db) > 0.0 ? 1.0 : -1.0);
            }
            sum += delta;
        }
    }

    // Multiply by 2 to match CPU double-counting behavior
    fragColor = vec4(2.0 * sum, 0.0, 0.0, 1.0);
}
)";

// Fragment shader: Jacobi iteration for r_solve
const char* rSolve_fsSrc = R"(
#version 130
uniform sampler2D grayTex;
uniform sampler2D dTex;
uniform int imageWidth;
uniform int imageHeight;
uniform int radius;
out vec4 fragColor;

void main() {
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);

    float dVal = texelFetch(dTex, ivec2(x, y), 0).r;
    float sum = 0.0;
    int count = 0;

    int xStart = max(0, x - radius);
    int xEnd = min(imageWidth - 1, x + radius);
    int yStart = max(0, y - radius);
    int yEnd = min(imageHeight - 1, y + radius);

    for (int xx = xStart; xx <= xEnd; xx++) {
        for (int yy = yStart; yy <= yEnd; yy++) {
            sum += texelFetch(grayTex, ivec2(xx, yy), 0).r;
            count++;
        }
    }

    fragColor = vec4((dVal + sum) / float(count), 0.0, 0.0, 1.0);
}
)";

GLuint compileShader(GLenum type, const char* source) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  GLint success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char log[1024];
    glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    std::cerr << "Shader compile error: " << log << std::endl;
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}

GLuint createProgram(const char* vsSrc, const char* fsSrc) {
  GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
  if (!vs || !fs) {
    if (vs) glDeleteShader(vs);
    if (fs) glDeleteShader(fs);
    return 0;
  }

  GLuint program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glBindAttribLocation(program, 0, "aPos");
  glLinkProgram(program);

  GLint success;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    char log[1024];
    glGetProgramInfoLog(program, sizeof(log), nullptr, log);
    std::cerr << "Program link error: " << log << std::endl;
    glDeleteProgram(program);
    program = 0;
  }

  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}

GLuint createTexture(int w, int h, GLenum internalFmt, GLenum fmt, GLenum type,
                     const void* data) {
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, internalFmt, w, h, 0, fmt, type, data);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  return tex;
}

void renderQuad() {
  glBindVertexArray(gVAO);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);
}

}  // namespace

bool GLCompute::init() {
  if (gInitialized) return true;

  if (!glfwInit()) {
    std::cerr << "GLCompute: Failed to initialize GLFW\n";
    return false;
  }

  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  gWindow = glfwCreateWindow(1, 1, "GLCompute", nullptr, nullptr);
  if (!gWindow) {
    std::cerr << "GLCompute: Failed to create GLFW window\n";
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(gWindow);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    std::cerr << "GLCompute: Failed to initialize GLEW\n";
    glfwDestroyWindow(gWindow);
    glfwTerminate();
    return false;
  }

  // Clear any GLEW-generated errors
  while (glGetError() != GL_NO_ERROR) {
  }

  std::cout << "GLCompute: OpenGL " << glGetString(GL_VERSION) << " ("
            << glGetString(GL_RENDERER) << ")\n";

  // Create full-screen quad VAO/VBO
  glGenVertexArrays(1, &gVAO);
  glGenBuffers(1, &gVBO);
  glBindVertexArray(gVAO);
  glBindBuffer(GL_ARRAY_BUFFER, gVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
  glBindVertexArray(0);

  // Compile all shader programs
  gCalcDProgram = createProgram(vsSrc, calcDNQ_fsSrc);
  gCalcDQProgram = createProgram(vsSrc, calcDQ_fsSrc);
  gRCalcDProgram = createProgram(vsSrc, rCalcD_fsSrc);
  gRSolveProgram = createProgram(vsSrc, rSolve_fsSrc);

  if (!gCalcDProgram || !gCalcDQProgram || !gRCalcDProgram ||
      !gRSolveProgram) {
    std::cerr << "GLCompute: Failed to compile one or more shader programs\n";
    cleanup();
    return false;
  }

  gInitialized = true;
  return true;
}

void GLCompute::cleanup() {
  if (gCalcDProgram) glDeleteProgram(gCalcDProgram);
  if (gCalcDQProgram) glDeleteProgram(gCalcDQProgram);
  if (gRCalcDProgram) glDeleteProgram(gRCalcDProgram);
  if (gRSolveProgram) glDeleteProgram(gRSolveProgram);
  gCalcDProgram = gCalcDQProgram = gRCalcDProgram = gRSolveProgram = 0;

  if (gVBO) glDeleteBuffers(1, &gVBO);
  if (gVAO) glDeleteVertexArrays(1, &gVAO);
  gVBO = gVAO = 0;

  if (gWindow) {
    glfwDestroyWindow(gWindow);
    gWindow = nullptr;
  }
  glfwTerminate();
  gInitialized = false;
}

bool GLCompute::isAvailable() { return gInitialized; }

std::vector<float> GLCompute::calcDNQ(const amy_lab* data, int N, int W, int H,
                                      float theta, float alpha) {
  // Pack Lab data into RGBA float texture
  std::vector<float> labData(4 * N);
  for (int i = 0; i < N; i++) {
    labData[4 * i + 0] = data[i].l;
    labData[4 * i + 1] = data[i].a;
    labData[4 * i + 2] = data[i].b;
    labData[4 * i + 3] = 0.0f;
  }

  GLuint labTex =
      createTexture(W, H, GL_RGBA32F, GL_RGBA, GL_FLOAT, labData.data());

  // Two accumulation textures for ping-pong
  std::vector<float> zeros(N, 0.0f);
  GLuint accumA =
      createTexture(W, H, GL_R32F, GL_RED, GL_FLOAT, zeros.data());
  GLuint accumB =
      createTexture(W, H, GL_R32F, GL_RED, GL_FLOAT, nullptr);

  GLuint fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);

  glViewport(0, 0, W, H);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  glUseProgram(gCalcDProgram);
  glUniform1i(glGetUniformLocation(gCalcDProgram, "labTex"), 0);
  glUniform1i(glGetUniformLocation(gCalcDProgram, "prevAccum"), 1);
  glUniform1i(glGetUniformLocation(gCalcDProgram, "imageWidth"), W);
  glUniform1i(glGetUniformLocation(gCalcDProgram, "imageHeight"), H);
  glUniform1f(glGetUniformLocation(gCalcDProgram, "theta"), theta);
  glUniform1f(glGetUniformLocation(gCalcDProgram, "alpha"), alpha);

  // Bind lab texture to unit 0
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, labTex);

  GLuint readTex = accumA;
  GLuint writeTex = accumB;
  int passCount = 0;

  for (int tile = 0; tile < N; tile += TILE_SIZE) {
    int tileEnd = std::min(tile + TILE_SIZE, N);

    // Attach write target
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           writeTex, 0);

    // Bind read source to unit 1
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, readTex);
    glActiveTexture(GL_TEXTURE0);

    glUniform1i(glGetUniformLocation(gCalcDProgram, "tileStart"), tile);
    glUniform1i(glGetUniformLocation(gCalcDProgram, "tileEnd"), tileEnd);

    renderQuad();

    std::swap(readTex, writeTex);
    passCount++;
    if (passCount % FLUSH_INTERVAL == 0) glFinish();
  }
  glFinish();

  // Read back result
  std::vector<float> result(N);
  glBindTexture(GL_TEXTURE_2D, readTex);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, result.data());

  // Cleanup
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
  glDeleteTextures(1, &labTex);
  glDeleteTextures(1, &accumA);
  glDeleteTextures(1, &accumB);

  return result;
}

std::vector<float> GLCompute::calcDQ(
    const amy_lab* data, int N, int W, int H,
    const std::vector<std::pair<amy_lab, int>>& qdata, float theta,
    float alpha) {
  // Pack Lab data into RGBA float texture
  std::vector<float> labData(4 * N);
  for (int i = 0; i < N; i++) {
    labData[4 * i + 0] = data[i].l;
    labData[4 * i + 1] = data[i].a;
    labData[4 * i + 2] = data[i].b;
    labData[4 * i + 3] = 0.0f;
  }

  GLuint labTex =
      createTexture(W, H, GL_RGBA32F, GL_RGBA, GL_FLOAT, labData.data());

  // Pack quantized color data: (L, a, b, count) per entry
  int numQ = static_cast<int>(qdata.size());
  std::vector<float> qTexData(4 * numQ);
  for (int i = 0; i < numQ; i++) {
    qTexData[4 * i + 0] = qdata[i].first.l;
    qTexData[4 * i + 1] = qdata[i].first.a;
    qTexData[4 * i + 2] = qdata[i].first.b;
    qTexData[4 * i + 3] = static_cast<float>(qdata[i].second);
  }

  GLuint qDataTex =
      createTexture(numQ, 1, GL_RGBA32F, GL_RGBA, GL_FLOAT, qTexData.data());

  // Output texture
  GLuint outTex = createTexture(W, H, GL_R32F, GL_RED, GL_FLOAT, nullptr);

  GLuint fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         outTex, 0);

  glViewport(0, 0, W, H);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  glUseProgram(gCalcDQProgram);
  glUniform1i(glGetUniformLocation(gCalcDQProgram, "labTex"), 0);
  glUniform1i(glGetUniformLocation(gCalcDQProgram, "qDataTex"), 1);
  glUniform1i(glGetUniformLocation(gCalcDQProgram, "imageWidth"), W);
  glUniform1i(glGetUniformLocation(gCalcDQProgram, "imageHeight"), H);
  glUniform1i(glGetUniformLocation(gCalcDQProgram, "numQColors"), numQ);
  glUniform1f(glGetUniformLocation(gCalcDQProgram, "theta"), theta);
  glUniform1f(glGetUniformLocation(gCalcDQProgram, "alpha"), alpha);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, labTex);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, qDataTex);

  renderQuad();
  glFinish();

  // Read back
  std::vector<float> result(N);
  glBindTexture(GL_TEXTURE_2D, outTex);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, result.data());

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
  glDeleteTextures(1, &labTex);
  glDeleteTextures(1, &qDataTex);
  glDeleteTextures(1, &outTex);

  return result;
}

std::vector<float> GLCompute::rCalcD(const amy_lab* data, int W, int H, int r,
                                     float theta, float alpha) {
  int N = W * H;

  // Pack Lab data
  std::vector<float> labData(4 * N);
  for (int i = 0; i < N; i++) {
    labData[4 * i + 0] = data[i].l;
    labData[4 * i + 1] = data[i].a;
    labData[4 * i + 2] = data[i].b;
    labData[4 * i + 3] = 0.0f;
  }

  GLuint labTex =
      createTexture(W, H, GL_RGBA32F, GL_RGBA, GL_FLOAT, labData.data());

  GLuint outTex = createTexture(W, H, GL_R32F, GL_RED, GL_FLOAT, nullptr);

  GLuint fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         outTex, 0);

  glViewport(0, 0, W, H);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  glUseProgram(gRCalcDProgram);
  glUniform1i(glGetUniformLocation(gRCalcDProgram, "labTex"), 0);
  glUniform1i(glGetUniformLocation(gRCalcDProgram, "imageWidth"), W);
  glUniform1i(glGetUniformLocation(gRCalcDProgram, "imageHeight"), H);
  glUniform1i(glGetUniformLocation(gRCalcDProgram, "radius"), r);
  glUniform1f(glGetUniformLocation(gRCalcDProgram, "theta"), theta);
  glUniform1f(glGetUniformLocation(gRCalcDProgram, "alpha"), alpha);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, labTex);

  renderQuad();
  glFinish();

  // Read back
  std::vector<float> result(N);
  glBindTexture(GL_TEXTURE_2D, outTex);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, result.data());

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
  glDeleteTextures(1, &labTex);
  glDeleteTextures(1, &outTex);

  return result;
}

void GLCompute::rSolve(float* grayData, const float* d, int W, int H, int r,
                       int iterations) {
  // Upload gray data and d values as textures
  GLuint grayTexA =
      createTexture(W, H, GL_R32F, GL_RED, GL_FLOAT, grayData);
  GLuint grayTexB =
      createTexture(W, H, GL_R32F, GL_RED, GL_FLOAT, nullptr);
  GLuint dTex = createTexture(W, H, GL_R32F, GL_RED, GL_FLOAT, d);

  GLuint fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);

  glViewport(0, 0, W, H);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  glUseProgram(gRSolveProgram);
  glUniform1i(glGetUniformLocation(gRSolveProgram, "grayTex"), 0);
  glUniform1i(glGetUniformLocation(gRSolveProgram, "dTex"), 1);
  glUniform1i(glGetUniformLocation(gRSolveProgram, "imageWidth"), W);
  glUniform1i(glGetUniformLocation(gRSolveProgram, "imageHeight"), H);
  glUniform1i(glGetUniformLocation(gRSolveProgram, "radius"), r);

  // Bind d texture to unit 1
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, dTex);

  GLuint readTex = grayTexA;
  GLuint writeTex = grayTexB;

  for (int k = 0; k < iterations; k++) {
    // Attach write target
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           writeTex, 0);

    // Bind previous iteration's gray values to unit 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, readTex);

    renderQuad();

    std::swap(readTex, writeTex);
    if ((k + 1) % FLUSH_INTERVAL == 0) glFinish();
  }
  glFinish();

  // Read back result
  glBindTexture(GL_TEXTURE_2D, readTex);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, grayData);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
  glDeleteTextures(1, &grayTexA);
  glDeleteTextures(1, &grayTexB);
  glDeleteTextures(1, &dTex);
}
