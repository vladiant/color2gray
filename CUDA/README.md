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
