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

## Benchmark

The table below compares all backends on the **same input image** using the **same
parameters**, reporting conversion time, peak host memory, and GPU utilization.

Benchmark configuration:

- Input image: `Lena.bmp` (512 × 512, 24-bit).
- Parameters: identical Color2Gray defaults across every backend
  (`theta = 0.523`, `alpha = 10`, `mu = 0` — full-image solve).
- Time: in-process conversion time reported by the program
  (`c2g completed in … milliseconds`, excludes image load/save).
- Memory: peak resident host memory (`/usr/bin/time -v`, maximum RSS).
- GPU util.: peak GPU compute utilization sampled with `nvidia-smi` during the run
  (not applicable for CPU-only backends).
- Machine: NVIDIA GeForce GTX 750 Ti (driver 580.196.07), single run.
- CUDA was built and run inside a `nvidia/cuda:12.5.1-devel-ubuntu22.04`
  container (compiled for `sm_50`), since no CUDA toolkit is installed on the
  host; all other backends ran natively.

| Backend     | Device   | Time (ms) | Peak memory (MB) | GPU util. (%) |
| ----------- | -------- | --------: | ---------------: | ------------: |
| BMP         | CPU      |   681 698 |             10.2 | N/A           |
| OpenCV      | CPU      |   718 382 |             92.7 | N/A           |
| CUDA        | GPU      |     7 962 |             74.7 | 100           |
| OpenCL      | GPU      |    11 347 |            108.6 | 100           |
| OpenGL      | GPU      |    12 927 |             71.6 | 100           |
| OpenGL CS   | GPU      |     8 871 |            126.3 | 100           |
| Vulkan      | GPU      |     7 162 |             72.2 | 100           |

> **Notes:**
> - The GPU backends offload the expensive `calc_d` step (an *O(N²)* pairwise
>   computation) to the device, so they run roughly 50–95× faster than the CPU
>   backends on this image; `complete_solve` and I/O remain on the host.
> - OpenCL was run with the `--gpu` flag to select the GPU device; without it the
>   OpenCL build falls back to the CPU path and performs like the BMP reference.
> - CUDA cannot be built on this host directly (the stock binary targets a newer
>   architecture than the GTX 750 Ti's `sm_50` and no CUDA toolkit is installed),
>   so it was compiled and run with the NVIDIA GPU-enabled Docker image below.
>   On a host with a matching toolkit it can be built and run natively.
> - Peak memory is host RSS; GPU-side allocations are not included.

### Running CUDA via Docker

The GTX 750 Ti (compute capability `5.0`, Maxwell) is still supported by the
CUDA 12.x toolkit (CUDA 13 dropped Maxwell). With Docker and the NVIDIA Container
Toolkit installed, the CUDA backend can be built and run from the repository root
without a host-side CUDA install:

```bash
docker run --rm --gpus all -v "$PWD":/work -w /work/CUDA \
  nvidia/cuda:12.5.1-devel-ubuntu22.04 bash -c '
    apt-get update -qq && apt-get install -y -qq cmake build-essential &&
    cmake -S . -B build-docker -DCMAKE_CUDA_ARCHITECTURES=50 &&
    cmake --build build-docker -j --target color2gray_bmp &&
    cp /work/BmpOnly/build/Lena.bmp build-docker/ &&
    ./build-docker/color2gray_bmp build-docker/Lena.bmp'
```

Set `-DCMAKE_CUDA_ARCHITECTURES` to match the target GPU (for example `50` for
Maxwell, `75` for Turing, `86` for Ampere).

## Paper

* Title: _Color2Gray: Salience-preserving color removal_
* Authors: _Amy Ashurst Gooch, Sven C. Olsen, Jack Tumblin, Bruce Gooch_
* Journal: _ACM Transactions on Graphics 24(3)_
* Date: _July 2005_
* Pages: _634-639_

## References

* [Color2Gray: Salience-Preserving Color Removal](https://users.cs.northwestern.edu/~ago820/color2gray/)
* [DOI:10.1145/1186822.1073241](http://dx.doi.org/10.1145/1186822.1073241)
