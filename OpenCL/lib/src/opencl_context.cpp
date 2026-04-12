#ifdef HAS_OPENCL

#include "opencl_context.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

// -------------------------------------------------------------------
// OpenCLContext implementation
// -------------------------------------------------------------------

OpenCLContext::OpenCLContext() {
  cl_int err = CL_SUCCESS;

  // --- platform ---
  cl_uint numPlatforms = 0;
  err = clGetPlatformIDs(0, nullptr, &numPlatforms);
  if (err != CL_SUCCESS || numPlatforms == 0) {
    std::cerr << "OpenCL: no platforms found.\n";
    return;
  }

  std::vector<cl_platform_id> platforms(numPlatforms);
  clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  mPlatform = platforms[0];

  // --- device (prefer GPU, fall back to CPU) ---
  cl_uint numDevices = 0;
  err = clGetDeviceIDs(mPlatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
  if (err != CL_SUCCESS || numDevices == 0) {
    err = clGetDeviceIDs(mPlatform, CL_DEVICE_TYPE_CPU, 0, nullptr,
                         &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
      std::cerr << "OpenCL: no devices found.\n";
      return;
    }
    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(mPlatform, CL_DEVICE_TYPE_CPU, numDevices, devices.data(),
                   nullptr);
    mDevice = devices[0];
    std::cout << "OpenCL: using CPU device.\n";
  } else {
    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(mPlatform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(),
                   nullptr);
    mDevice = devices[0];
    std::cout << "OpenCL: using GPU device.\n";
  }

  // Print device name
  char deviceName[256] = {};
  clGetDeviceInfo(mDevice, CL_DEVICE_NAME, sizeof(deviceName), deviceName,
                  nullptr);
  std::cout << "OpenCL device: " << deviceName << '\n';

  // --- context ---
  mContext = clCreateContext(nullptr, 1, &mDevice, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL: failed to create context (" << err << ").\n";
    return;
  }

  // --- command queue (OpenCL 1.2 compatible) ---
  cl_command_queue_properties props = 0;
  mQueue = clCreateCommandQueue(mContext, mDevice, props, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL: failed to create command queue (" << err << ").\n";
    cleanup();
    return;
  }

  mValid = true;
}

OpenCLContext::~OpenCLContext() { cleanup(); }

void OpenCLContext::cleanup() {
  if (mProgram) clReleaseProgram(mProgram);
  if (mQueue) clReleaseCommandQueue(mQueue);
  if (mContext) clReleaseContext(mContext);
  mProgram = nullptr;
  mQueue = nullptr;
  mContext = nullptr;
  mValid = false;
}

bool OpenCLContext::buildProgram(const std::string& source) {
  cl_int err = CL_SUCCESS;

  if (mProgram) {
    clReleaseProgram(mProgram);
    mProgram = nullptr;
  }

  const char* src = source.c_str();
  size_t len = source.size();
  mProgram = clCreateProgramWithSource(mContext, 1, &src, &len, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL: failed to create program (" << err << ").\n";
    return false;
  }

  err = clBuildProgram(mProgram, 1, &mDevice, "-cl-std=CL1.2", nullptr,
                       nullptr);
  if (err != CL_SUCCESS) {
    size_t logSize = 0;
    clGetProgramBuildInfo(mProgram, mDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &logSize);
    std::string buildLog(logSize, '\0');
    clGetProgramBuildInfo(mProgram, mDevice, CL_PROGRAM_BUILD_LOG, logSize,
                          buildLog.data(), nullptr);
    std::cerr << "OpenCL build error:\n" << buildLog << '\n';
    clReleaseProgram(mProgram);
    mProgram = nullptr;
    return false;
  }

  return true;
}

cl_kernel OpenCLContext::createKernel(const char* name) const {
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(mProgram, name, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL: failed to create kernel '" << name << "' (" << err
              << ").\n";
    return nullptr;
  }
  return kernel;
}

// -------------------------------------------------------------------
// Utility
// -------------------------------------------------------------------

std::string readKernelSource(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Failed to open kernel file: " << path << '\n';
    return {};
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

#endif  // HAS_OPENCL
