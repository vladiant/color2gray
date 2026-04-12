#include "color_image.hpp"

#include <cmath>
#include <iostream>
#include <map>

#include "utils.hpp"

ColorImage::ColorImage(float a_theta, float a_alpha, bool a_quantize)
    : mTheta{a_theta}, mAlpha{a_alpha}, mQuantize{a_quantize} {}

std::vector<float> ColorImage::calc_d() {
  std::vector<float> d(mN);
  if (mQuantize) {
    calc_d_q(mN, mData.data(), mQdata, mTheta, mAlpha, d.data());
  } else {
    calc_d_nq(mN, mData.data(), mTheta, mAlpha, d.data());
  }
  return d;
}

std::vector<float> ColorImage::r_calc_d(int r) {
  std::vector<float> d(mN);
  int i = 0;
  for (i = 0; i < mN; i++) d[i] = 0;

  int x = 0, y = 0;
  for (x = 0; x < mW; x++)
    for (y = 0; y < mH; y++) {
      int xx = 0, yy = 0;

      i = x + y * mW;

      for (xx = x - r; xx <= x + r; xx++) {
        if (xx < 0 || xx >= mW) continue;
        for (yy = y - r; yy <= y + r; yy++) {
          if (yy >= mH || yy < 0) continue;
          int j = xx + yy * mW;
          float delta = calc_delta(i, j, mData.data(), mTheta, mAlpha);
          d[i] += delta;
          d[j] -= delta;
        }
      }
    }
  return d;
}

void ColorImage::load(const std::vector<uint8_t>& imgData, const int width,
                      const int height) {
  using sven::rgb;

  mW = width;
  mH = height;

  mN = mW * mH;
  std::cout << "image loaded, w: " << mW << ", y: " << mH << ".\n";

  mData.resize(mN);

  for (int i = 0; i < mN; i++) {
    mData[i] =
        amy_lab(rgb(imgData[3 * i], imgData[3 * i + 1], imgData[3 * i + 2]));
  }
}

void ColorImage::load_quant_data(const std::vector<uint8_t>& imgData,
                                 const int width, const int height) {
  using sven::rgb;

  mW = width;
  mH = height;

  std::vector<rgb> colors;

  mData.resize(mN);

  for (int i = 0; i < mN; i++) {
    colors.emplace_back(imgData[3 * i], imgData[3 * i + 1], imgData[3 * i + 2]);
    mData[i] =
        amy_lab(rgb(imgData[3 * i], imgData[3 * i + 1], imgData[3 * i + 2]));
  }

  mN = mW * mH;
  std::cout << "quantized image loaded, w: " << mW << ", y: " << mH << ".\n";

  mQdata.clear();

  using namespace std;

  map<rgb, int> q;
  map<rgb, int>::iterator r;
  for (int i = 0; i < mN; i++) {
    r = q.find(colors[i]);
    if (r == q.end())
      q[colors[i]] = 1;
    else
      r->second++;
  }

  std::cout << "quantized image appears to use " << q.size() << " colors.\n";
  mQdata.resize(q.size());
  int i = 0;
  for (i = 0, r = q.begin(); r != q.end(); ++r, i++) {
    mQdata[i] = amy_lab_int(amy_lab(r->first), r->second);
  }
}

#ifdef HAS_OPENCL

#include "opencl_context.hpp"

std::vector<float> ColorImage::getPackedLAB() const {
  std::vector<float> packed(3 * mN);
  for (int i = 0; i < mN; i++) {
    packed[3 * i]     = mData[i].l;
    packed[3 * i + 1] = mData[i].a;
    packed[3 * i + 2] = mData[i].b;
  }
  return packed;
}

void ColorImage::load_ocl(const std::vector<uint8_t>& imgData, const int width,
                           const int height, OpenCLContext& ocl) {
  using sven::rgb;

  mW = width;
  mH = height;
  mN = mW * mH;

  std::cout << "image loaded (OCL), w: " << mW << ", y: " << mH << ".\n";

  // Build the color_convert kernel
  const std::string src = readKernelSource("kernels/color_convert.cl");
  if (src.empty() || !ocl.buildProgram(src)) {
    std::cerr << "OpenCL: color_convert kernel build failed, falling back to "
                 "CPU.\n";
    load(imgData, width, height);
    return;
  }

  cl_kernel kernel = ocl.createKernel("rgb_to_lab");
  if (!kernel) {
    load(imgData, width, height);
    return;
  }

  cl_int err = CL_SUCCESS;
  const size_t rgbBytes = 3 * mN * sizeof(unsigned char);
  const size_t labBytes = 3 * mN * sizeof(float);

  cl_mem rgbBuf = clCreateBuffer(ocl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 rgbBytes, const_cast<uint8_t*>(imgData.data()), &err);
  cl_mem labBuf = clCreateBuffer(ocl.context(), CL_MEM_WRITE_ONLY, labBytes,
                                 nullptr, &err);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &rgbBuf);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &labBuf);
  clSetKernelArg(kernel, 2, sizeof(int), &mN);

  size_t globalSize = static_cast<size_t>(mN);
  err = clEnqueueNDRangeKernel(ocl.queue(), kernel, 1, nullptr, &globalSize,
                               nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL: rgb_to_lab kernel launch failed (" << err << ").\n";
    clReleaseMemObject(rgbBuf);
    clReleaseMemObject(labBuf);
    clReleaseKernel(kernel);
    load(imgData, width, height);
    return;
  }

  std::vector<float> labFlat(3 * mN);
  clEnqueueReadBuffer(ocl.queue(), labBuf, CL_TRUE, 0, labBytes,
                      labFlat.data(), 0, nullptr, nullptr);

  mData.resize(mN);
  for (int i = 0; i < mN; i++) {
    mData[i].l = labFlat[3 * i];
    mData[i].a = labFlat[3 * i + 1];
    mData[i].b = labFlat[3 * i + 2];
  }

  clReleaseMemObject(rgbBuf);
  clReleaseMemObject(labBuf);
  clReleaseKernel(kernel);
}

std::vector<float> ColorImage::calc_d_ocl(OpenCLContext& ocl) {
  // Concatenate all kernel sources needed
  const std::string src = readKernelSource("kernels/calc_delta.cl");
  if (src.empty() || !ocl.buildProgram(src)) {
    std::cerr << "OpenCL: calc_delta kernel build failed, falling back to CPU.\n";
    return calc_d();
  }

  const char* kernelName = mQuantize ? "calc_d_q_kernel" : "calc_d_nq_kernel";
  cl_kernel kernel = ocl.createKernel(kernelName);
  if (!kernel) return calc_d();

  cl_int err = CL_SUCCESS;
  const auto labPacked = getPackedLAB();
  const size_t labBytes = labPacked.size() * sizeof(float);
  const size_t dBytes = mN * sizeof(float);

  cl_mem labBuf = clCreateBuffer(ocl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 labBytes, const_cast<float*>(labPacked.data()), &err);
  cl_mem dBuf = clCreateBuffer(ocl.context(), CL_MEM_WRITE_ONLY, dBytes,
                                nullptr, &err);

  if (mQuantize && !mQdata.empty()) {
    // Prepare quantized data buffers
    const int Q = static_cast<int>(mQdata.size());
    std::vector<float> qlabPacked(3 * Q);
    std::vector<int> qcounts(Q);
    for (int p = 0; p < Q; p++) {
      qlabPacked[3 * p]     = mQdata[p].first.l;
      qlabPacked[3 * p + 1] = mQdata[p].first.a;
      qlabPacked[3 * p + 2] = mQdata[p].first.b;
      qcounts[p] = mQdata[p].second;
    }

    cl_mem qlabBuf = clCreateBuffer(
        ocl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        qlabPacked.size() * sizeof(float),
        const_cast<float*>(qlabPacked.data()), &err);
    cl_mem qcountBuf = clCreateBuffer(
        ocl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        qcounts.size() * sizeof(int), const_cast<int*>(qcounts.data()), &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &labBuf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &qlabBuf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &qcountBuf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &dBuf);
    clSetKernelArg(kernel, 4, sizeof(float), &mTheta);
    clSetKernelArg(kernel, 5, sizeof(float), &mAlpha);
    clSetKernelArg(kernel, 6, sizeof(int), &mN);
    clSetKernelArg(kernel, 7, sizeof(int), &Q);

    size_t globalSize = static_cast<size_t>(mN);
    err = clEnqueueNDRangeKernel(ocl.queue(), kernel, 1, nullptr, &globalSize,
                                 nullptr, 0, nullptr, nullptr);

    clReleaseMemObject(qlabBuf);
    clReleaseMemObject(qcountBuf);
  } else {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &labBuf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dBuf);
    clSetKernelArg(kernel, 2, sizeof(float), &mTheta);
    clSetKernelArg(kernel, 3, sizeof(float), &mAlpha);
    clSetKernelArg(kernel, 4, sizeof(int), &mN);

    size_t globalSize = static_cast<size_t>(mN);
    err = clEnqueueNDRangeKernel(ocl.queue(), kernel, 1, nullptr, &globalSize,
                                 nullptr, 0, nullptr, nullptr);
  }

  std::vector<float> d(mN);
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL: " << kernelName << " launch failed (" << err
              << "), falling back to CPU.\n";
    clReleaseMemObject(labBuf);
    clReleaseMemObject(dBuf);
    clReleaseKernel(kernel);
    return calc_d();
  }

  clEnqueueReadBuffer(ocl.queue(), dBuf, CL_TRUE, 0, dBytes, d.data(), 0,
                      nullptr, nullptr);

  clReleaseMemObject(labBuf);
  clReleaseMemObject(dBuf);
  clReleaseKernel(kernel);
  return d;
}

std::vector<float> ColorImage::r_calc_d_ocl(int r, OpenCLContext& ocl) {
  const std::string src = readKernelSource("kernels/calc_delta.cl");
  if (src.empty() || !ocl.buildProgram(src)) {
    std::cerr << "OpenCL: r_calc_d kernel build failed, falling back to CPU.\n";
    return r_calc_d(r);
  }

  cl_kernel kernel = ocl.createKernel("r_calc_d_kernel");
  if (!kernel) return r_calc_d(r);

  cl_int err = CL_SUCCESS;
  const auto labPacked = getPackedLAB();
  const size_t labBytes = labPacked.size() * sizeof(float);
  const size_t dBytes = mN * sizeof(float);

  cl_mem labBuf = clCreateBuffer(ocl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 labBytes, const_cast<float*>(labPacked.data()), &err);
  cl_mem dBuf = clCreateBuffer(ocl.context(), CL_MEM_WRITE_ONLY, dBytes,
                                nullptr, &err);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &labBuf);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &dBuf);
  clSetKernelArg(kernel, 2, sizeof(float), &mTheta);
  clSetKernelArg(kernel, 3, sizeof(float), &mAlpha);
  clSetKernelArg(kernel, 4, sizeof(int), &mW);
  clSetKernelArg(kernel, 5, sizeof(int), &mH);
  clSetKernelArg(kernel, 6, sizeof(int), &r);

  size_t globalSize[2] = {static_cast<size_t>(mW), static_cast<size_t>(mH)};
  err = clEnqueueNDRangeKernel(ocl.queue(), kernel, 2, nullptr, globalSize,
                               nullptr, 0, nullptr, nullptr);

  std::vector<float> d(mN);
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL: r_calc_d_kernel launch failed (" << err
              << "), falling back to CPU.\n";
    clReleaseMemObject(labBuf);
    clReleaseMemObject(dBuf);
    clReleaseKernel(kernel);
    return r_calc_d(r);
  }

  clEnqueueReadBuffer(ocl.queue(), dBuf, CL_TRUE, 0, dBytes, d.data(), 0,
                      nullptr, nullptr);

  clReleaseMemObject(labBuf);
  clReleaseMemObject(dBuf);
  clReleaseKernel(kernel);
  return d;
}

#endif  // HAS_OPENCL
