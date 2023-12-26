// -----------------------------------------------------
// bitmap.h
//
// header file for MS bitmap format
// -----------------------------------------------------

#pragma once

#include <cstdint>
#include <string>
#include <vector>

// global I/O routines
std::vector<uint8_t> readBMP(const std::string& fname, int& width, int& height);

void writeBMP(const std::string& iname, int width, int height,
              const std::vector<uint8_t>& data);

