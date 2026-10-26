# color2gray

This repository contains seven implementations of the Color2Gray salience-preserving grayscale conversion algorithm.

## Implementations

- BMP — reference implementation using BMP image I/O and a CPU-based solver.
- OpenCV — implementation using OpenCV for image loading, processing, and output.
- CUDA — NVIDIA GPU implementation using CUDA kernels for parallel computation.
- OpenCL — cross-platform GPU implementation using OpenCL kernels.
- OpenGL — OpenGL-based implementation that renders the grayscale conversion on the GPU.
- OpenGL CS — OpenGL compute-shader implementation (`OpenGlComputeShaders/`).
- Vulkan — Vulkan compute-shader implementation for GPU acceleration.

## Project layout

- `BmpOnly/`
- `OpenCV/`
- `CUDA/`
- `OpenCL/`
- `OpenGL/`
- `OpenGlComputeShaders/`
- `Vulkan/`

Each directory contains a separate C++/GPU implementation of the same algorithm, allowing comparison across CPU, OpenCV, CUDA, OpenCL, OpenGL, and Vulkan backends.

## Correctness check

All seven implementations should be cross-validated against the same input images and compared pixel-by-pixel after conversion to grayscale.

Recommended validation procedure:

1. Run each implementation on the same reference image set.
2. Save each output as an equivalent grayscale BMP image.
3. Compare the resulting images with a strict or epsilon-based checksum:
   - exact match is expected for 8-bit grayscale output when the same algorithm and rounding policy are used;
   - otherwise, use a small tolerance such as `|a - b| <= 1` or an absolute/relative error threshold (for example, `1e-4` in floating-point pipelines).
4. Report maximum absolute difference, mean squared error, and a pass/fail result for each implementation pair.
5. Treat any deviation beyond the chosen epsilon as a correctness issue and investigate rounding, quantization, or format-conversion differences.

This check ensures the implementations remain behaviorally consistent even when they use different hardware and programming models.

## Paper

* Title: _Color2Gray: Salience-preserving color removal_
* Authors: _Amy Ashurst Gooch, Sven C. Olsen, Jack Tumblin, Bruce Gooch_
* Journal: _ACM Transactions on Graphics 24(3)_
* Date: _July 2005_
* Pages: _634-639_

## References

* [Color2Gray: Salience-Preserving Color Removal](https://users.cs.northwestern.edu/~ago820/color2gray/)
* [DOI:10.1145/1186822.1073241](http://dx.doi.org/10.1145/1186822.1073241)
