#pragma once

#include <cstdint>
#include <vector>

void compare_images(const std::vector<uint8_t>& lhs,
                    const std::vector<uint8_t>& rhs);

/// Compare images allowing a per-pixel tolerance (for GPU floating-point
/// differences).
void compare_images_fuzzy(const std::vector<uint8_t>& lhs,
                          const std::vector<uint8_t>& rhs,
                          int tolerance = 1);
