#pragma once

#ifdef HAS_OPENCL

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <string>
#include <vector>

/// RAII wrapper for OpenCL platform, device, context, queue, and compiled
/// program. Prefers GPU device; falls back to CPU.
class OpenCLContext {
 public:
  OpenCLContext();
  ~OpenCLContext();

  OpenCLContext(const OpenCLContext&) = delete;
  OpenCLContext& operator=(const OpenCLContext&) = delete;

  /// Returns true if the context was initialized successfully.
  bool isValid() const { return mValid; }

  /// Build a program from a .cl source string. Returns true on success.
  bool buildProgram(const std::string& source);

  /// Create a kernel by name from the compiled program.
  cl_kernel createKernel(const char* name) const;

  cl_context context() const { return mContext; }
  cl_command_queue queue() const { return mQueue; }
  cl_device_id device() const { return mDevice; }

 private:
  bool mValid{false};
  cl_platform_id mPlatform{nullptr};
  cl_device_id mDevice{nullptr};
  cl_context mContext{nullptr};
  cl_command_queue mQueue{nullptr};
  cl_program mProgram{nullptr};

  void cleanup();
};

/// Read the contents of a text file into a string.
std::string readKernelSource(const std::string& path);

#endif  // HAS_OPENCL
