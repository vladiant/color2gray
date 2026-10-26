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

## Paper

* Title: _Color2Gray: Salience-preserving color removal_
* Authors: _Amy Ashurst Gooch, Sven C. Olsen, Jack Tumblin, Bruce Gooch_
* Journal: _ACM Transactions on Graphics 24(3)_
* Date: _July 2005_
* Pages: _634-639_

## References

* [Color2Gray: Salience-Preserving Color Removal](https://users.cs.northwestern.edu/~ago820/color2gray/)
* [DOI:10.1145/1186822.1073241](http://dx.doi.org/10.1145/1186822.1073241)
