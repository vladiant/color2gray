#include "gray_image.hpp"

#include <iostream>

#include "bitmap.hpp"

GrayImage::GrayImage(const ColorImage &s)
    : mData(s.getN()), mW(s.getW()), mH(s.getH()), mN(s.getN()) {
  const auto &data = s.getData();
  for (int i = 0; i < mN; i++) mData[i] = data[i].l;
}

void GrayImage::r_solve(const std::vector<float> &d, int r) {
  constexpr int iters = 30;
  int k = 0, x = 0, y = 0;

  for (k = 0; k < iters; k++) {
    // std::cout << "iter " << k << "\n";

    // perform a Gauss-Seidel relaxation.
    for (x = 0; x < mW; x++)
      for (y = 0; y < mH; y++) {
        float sum = 0;
        int count = 0;
        int xx = 0, yy = 0;

        for (xx = x - r; xx <= x + r; xx++) {
          if (xx < 0 || xx >= mW) continue;
          for (yy = y - r; yy <= y + r; yy++) {
            if (yy >= mH || yy < 0) continue;
            sum += mData[xx + yy * mW];
            count++;
          }
        }

        mData[x + y * mW] = (d[x + mW * y] + sum) / (float)count;
      }
  }
}

void GrayImage::complete_solve(const std::vector<float> &d) {
  for (int i = 1; i < mN; i++) {
    mData[i] = d[i] - d[i - 1] + mN * mData[i - 1];
    mData[i] /= (float)mN;
  }
}

void GrayImage::post_solve(const ColorImage &s) {
  float error = 0;
  const auto &data = s.getData();
  for (int i = 0; i < mN; i++) error += mData[i] - data[i].l;
  error /= mN;
  for (int i = 0; i < mN; i++) mData[i] = mData[i] - error;
}

std::vector<uint8_t> GrayImage::save(const char *fname) const {
  std::vector<uint8_t> bmpData;
  bmpData.reserve(3 * mN);
  for (int i = 0; i < mN; i++) {
    const sven::rgb rval = amy_lab(mData[i], 0, 0).to_rgb();
    bmpData.push_back(rval.r);
    bmpData.push_back(rval.g);
    bmpData.push_back(rval.b);
  }

  if (fname) {
    writeBMP(fname, mW, mH, bmpData);
  }

  return bmpData;
}

std::vector<uint8_t> GrayImage::saveColor(const char *fname,
                                          const ColorImage &source) const {
  const auto &data = source.getData();

  std::vector<uint8_t> bmpData;
  bmpData.reserve(3 * mN);
  for (int i = 0; i < mN; i++) {
    const sven::rgb rval = amy_lab(mData[i], (data[i]).a, (data[i]).b).to_rgb();
    bmpData.push_back(rval.r);
    bmpData.push_back(rval.g);
    bmpData.push_back(rval.b);
  }

  if (fname) {
    writeBMP(fname, mW, mH, bmpData);
  }

  return bmpData;
}

#ifdef HAS_OPENCL

#include "opencl_context.hpp"

void GrayImage::r_solve_ocl(const std::vector<float>& d, int r,
                             OpenCLContext& ocl) {
  // Jacobi iteration: replaces Gauss-Seidel for GPU parallelism.
  // Uses double buffering — read from src, write to dst, swap each iteration.
  constexpr int iters = 50;  // More iterations than Gauss-Seidel (30)

  const std::string src = readKernelSource("kernels/solver.cl");
  if (src.empty() || !ocl.buildProgram(src)) {
    std::cerr << "OpenCL: solver kernel build failed, falling back to CPU.\n";
    r_solve(d, r);
    return;
  }

  cl_kernel kernel = ocl.createKernel("jacobi_iteration");
  if (!kernel) {
    r_solve(d, r);
    return;
  }

  cl_int err = CL_SUCCESS;
  const size_t dataBytes = mN * sizeof(float);

  // Upload d (constant) and initial mData
  cl_mem dBuf = clCreateBuffer(ocl.context(),
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               dataBytes, const_cast<float*>(d.data()), &err);
  cl_mem bufA = clCreateBuffer(ocl.context(),
                               CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               dataBytes, mData.data(), &err);
  cl_mem bufB = clCreateBuffer(ocl.context(), CL_MEM_READ_WRITE, dataBytes,
                               nullptr, &err);

  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL: buffer creation failed (" << err
              << "), falling back to CPU.\n";
    if (dBuf) clReleaseMemObject(dBuf);
    if (bufA) clReleaseMemObject(bufA);
    if (bufB) clReleaseMemObject(bufB);
    clReleaseKernel(kernel);
    r_solve(d, r);
    return;
  }

  size_t globalSize[2] = {static_cast<size_t>(mW), static_cast<size_t>(mH)};

  // Set constant args: d, W, H, r
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &dBuf);
  clSetKernelArg(kernel, 3, sizeof(int), &mW);
  clSetKernelArg(kernel, 4, sizeof(int), &mH);
  clSetKernelArg(kernel, 5, sizeof(int), &r);

  cl_mem currentSrc = bufA;
  cl_mem currentDst = bufB;

  for (int k = 0; k < iters; k++) {
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &currentSrc);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &currentDst);

    err = clEnqueueNDRangeKernel(ocl.queue(), kernel, 2, nullptr, globalSize,
                                 nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::cerr << "OpenCL: jacobi_iteration launch failed at iter " << k
                << " (" << err << "), falling back to CPU.\n";
      clReleaseMemObject(dBuf);
      clReleaseMemObject(bufA);
      clReleaseMemObject(bufB);
      clReleaseKernel(kernel);
      r_solve(d, r);
      return;
    }

    // Swap buffers
    cl_mem tmp = currentSrc;
    currentSrc = currentDst;
    currentDst = tmp;
  }

  // After iters iterations, result is in currentSrc (last written dst, now src)
  clEnqueueReadBuffer(ocl.queue(), currentSrc, CL_TRUE, 0, dataBytes,
                      mData.data(), 0, nullptr, nullptr);

  clReleaseMemObject(dBuf);
  clReleaseMemObject(bufA);
  clReleaseMemObject(bufB);
  clReleaseKernel(kernel);
}

#endif  // HAS_OPENCL