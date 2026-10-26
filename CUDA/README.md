# color2gray (CUDA)

A GPU-accelerated implementation of the **color2gray** algorithm by Amy Gooch et al., which converts colour images to greyscale while preserving perceptual contrast that would otherwise be lost.

The algorithm maps each pixel from RGB → CIELAB, computes a signed perceptual difference (Δ) between pixels, and solves a system that assigns greyscale values maximising those differences.

## Algorithm overview

1. **RGB → CIELAB** – every pixel is converted to the L\*a\*b\* colour space.
2. **Delta computation** – a signed perceptual difference Δ(i, j) is calculated for each pixel pair using luminance (L\*) and chrominance (a\*, b\*), controlled by parameters α and θ.
3. **Solve** – a linear system is solved to find greyscale values that satisfy the pairwise differences:
   - *Full solve* (`--r 0`): closed-form O(N²) solution over all pixel pairs.
   - *Neighbourhood solve* (`--r N`): iterative Gauss-Seidel relaxation over an r×r neighbourhood.
4. **Post-solve** – greyscale values are shifted to match the mean luminance of the original image.

## CUDA acceleration

All compute-intensive steps run on the GPU:

| Function | Kernel | Notes |
|---|---|---|
| `calc_d` (full) | `k_calc_d_nq` | Each thread sums Δ(i,j) over all N pixels |
| `calc_d` (quantised) | `k_calc_d_q` | Each thread sums weighted Δ over Q cluster centres |
| `r_calc_d` | `k_r_calc_d` | Each thread sums Δ over an r×r neighbourhood |
| `complete_solve` | `k_complete_solve` | Parallel closed-form: `g[i] = g[0] + (d[i]−d[0]) / N` |
| `r_solve` | `k_r_solve_rbgs` | Red-Black Gauss-Seidel, 30 sweeps |
| `post_solve` | `k_reduce_error` + `k_post_solve_apply` | Parallel reduction then scalar subtract |

## Requirements

| Dependency | Minimum version |
|---|---|
| CMake | 3.20 |
| C++ compiler | C++20 |
| CUDA Toolkit | 11.0 (tested with 12.0) |
| NVIDIA GPU | Any CUDA-capable device |

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

By default CMake targets the GPU present on the build machine (`native`). To target a specific compute capability, pass it explicitly:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86  # e.g. Ampere
cmake --build build -j$(nproc)
```

Common architecture values: `50` (Maxwell), `61` (Pascal), `75` (Turing), `86` (Ampere), `89` (Ada).

## Building and running with Docker

If the host has no CUDA toolkit installed — or its toolkit no longer supports the
target GPU (for example CUDA 13 dropped Maxwell / `sm_50`) — the backend can be
built and run inside an NVIDIA CUDA container instead. This requires:

- [Docker](https://docs.docker.com/engine/install/)
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (so `--gpus all` exposes the GPU to the container)

Pick a `*-devel` image whose CUDA version still supports the GPU's compute
capability (CUDA 12.x supports Maxwell `sm_50`; CUDA 13 does not).

First, verify the GPU is visible from a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.5.1-base-ubuntu22.04 nvidia-smi -L
```

Then build and run the backend from this directory (`CUDA/`). Set
`-DCMAKE_CUDA_ARCHITECTURES` to match the target GPU (`50` for the GTX 750 Ti):

```bash
docker run --rm --gpus all -v "$PWD":/work -w /work \
  nvidia/cuda:12.5.1-devel-ubuntu22.04 bash -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq cmake build-essential &&
    cmake -S . -B build-docker -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CUDA_ARCHITECTURES=50 &&
    cmake --build build-docker -j"$(nproc)" --target color2gray_bmp &&
    ./build-docker/color2gray_bmp your_image.bmp'
```

Notes:

- The `-v "$PWD":/work` mount makes the build output and any result BMPs appear
  on the host under `CUDA/build-docker/`. Files created by the container are
  owned by `root`; use `sudo` (or the cleanup command below) to remove them.
- To get an interactive shell for repeated builds, drop the `bash -c '…'` and run
  `… nvidia/cuda:12.5.1-devel-ubuntu22.04 bash`.

Remove the root-owned build directory when finished:

```bash
docker run --rm -v "$PWD":/work ubuntu:22.04 rm -rf /work/build-docker
```

## Usage

```
color2gray_bmp image.bmp [options]

Options:
  --help           Print help message
  --theta <deg>    Theta angle in degrees (default: ~30°)
  --alpha <val>    Alpha – controls chrominance weight (default: 10)
  --r <val>        Neighbourhood radius µ; 0 = full solve (default: 0)
  --q <colors>     Enable k-means quantisation with <colors> clusters
```

### Examples

Full solve (default):
```bash
./build/color2gray_bmp photo.bmp --alpha 10 --theta 30
```

Neighbourhood solve (faster on large images):
```bash
./build/color2gray_bmp photo.bmp --r 5 --alpha 10
```

Quantised neighbourhood solve:
```bash
./build/color2gray_bmp photo.bmp --r 5 --q 16
```

Outputs are written alongside the input file:
- `photo_c2g.bmp` – greyscale result
- `photo_c2g_color.bmp` – greyscale luminance with original chrominance (false-colour diagnostic)

## Running tests

```bash
cd build
ctest --output-on-failure
```

## Project structure

```
.
├── CMakeLists.txt
├── main.cpp                    # CLI entry point
├── argspp/                     # Argument parsing library
├── lib/
│   ├── include/
│   │   ├── amy_lab.hpp         # CIELAB colour type
│   │   ├── amy_xyz.hpp         # CIE XYZ colour type
│   │   ├── bitmap.hpp          # BMP read/write
│   │   ├── color_image.hpp     # Source image + delta computation
│   │   ├── cuda_kernels.hpp    # CUDA wrapper declarations
│   │   ├── gray_image.hpp      # Output image + solver
│   │   ├── rgb.hpp             # sRGB type
│   │   ├── time_bench.hpp      # RAII timer
│   │   └── utils.hpp           # k-means quantisation + Δ helpers
│   └── src/
│       ├── cuda_kernels.cu     # CUDA kernels (all GPU work)
│       ├── color_image.cpp
│       ├── gray_image.cpp
│       └── ...
└── test/
    ├── doctest/                # doctest header-only framework
    ├── test_color2gray.cpp     # Integration tests
    ├── test_data.cpp/hpp       # Golden reference data
    ├── test_utils.cpp/hpp      # Test helpers
    └── gen_expected.cpp        # Regenerate golden data from GPU output
```

## References

- A. Gooch, S. Olsen, J. Tumblin, B. Gooch, *"Color2Gray: Salience-Preserving Color Removal"*, SIGGRAPH 2005. [https://doi.org/10.1145/1186822.1073308](https://doi.org/10.1145/1186822.1073308)
