// ---------------------------------------------------------------------------
// bitmap.cpp
//
// handle MS bitmap I/O. For portability, we don't use the data structure
// defined in Windows.h. However, there is some strange thing, the size of our
// structure is different from what it should be though we define it in the
// same way as MS did. So, there is a hack, we use the hardcoded constant, 14,
// instead of the sizeof to calculate the size of the structure.  You are not
// supposed to worry about this part. However, I will appreciate if you find
// out the reason and let me know. Thanks.
// ---------------------------------------------------------------------------

#include "bitmap.hpp"

#include <cstring>
#include <fstream>
#include <iostream>

namespace {
constexpr auto BMP_BI_RGB = 0L;

using BMP_WORD = uint16_t;
using BMP_DWORD = uint32_t;
using BMP_LONG = int32_t;

using BMP_BITMAPFILEHEADER = struct {
  BMP_WORD bfType;
  BMP_DWORD bfSize;
  BMP_WORD bfReserved1;
  BMP_WORD bfReserved2;
  BMP_DWORD bfOffBits;
};

using BMP_BITMAPINFOHEADER = struct {
  BMP_DWORD biSize;
  BMP_LONG biWidth;
  BMP_LONG biHeight;
  BMP_WORD biPlanes;
  BMP_WORD biBitCount;
  BMP_DWORD biCompression;
  BMP_DWORD biSizeImage;
  BMP_LONG biXPelsPerMeter;
  BMP_LONG biYPelsPerMeter;
  BMP_DWORD biClrUsed;
  BMP_DWORD biClrImportant;
};
}  // namespace

BMP_BITMAPFILEHEADER bmfh;
BMP_BITMAPINFOHEADER bmih;

template <class T>
void swapBytes(T* val) {
  static int typeSize;
  static char *start, *end;
  typeSize = sizeof(T);

  start = (char*)val;
  end = start + typeSize - 1;

  while (start < end) {
    *start ^= *end ^= *start ^= *end;
    start++;
    end--;
  }
}

// Bitmap data returned is (R,G,B) tuples in row-major order.
std::vector<uint8_t> readBMP(const std::string& fname, int& width,
                             int& height) {
  std::fstream file(fname, std::fstream::in | std::fstream::binary);
  BMP_DWORD pos = 0;

  if (!file) return {};

  //	I am doing file.read(reinterpret_cast<char*>(&bmfh),
  // sizeof(BMP_BITMAPFILEHEADER)) in a
  // safe way. :}
  file.read(reinterpret_cast<char*>(&(bmfh.bfType)), 2);
  file.read(reinterpret_cast<char*>(&(bmfh.bfSize)), 4);
  file.read(reinterpret_cast<char*>(&(bmfh.bfReserved1)), 2);
  file.read(reinterpret_cast<char*>(&(bmfh.bfReserved2)), 2);
  file.read(reinterpret_cast<char*>(&(bmfh.bfOffBits)), 4);

  pos = bmfh.bfOffBits;

  file.read(reinterpret_cast<char*>(&bmih), sizeof(BMP_BITMAPINFOHEADER));

  // error checking
  if (bmfh.bfType != 0x4d42) {  // "BM" actually
    return {};
  }
  if (bmih.biBitCount != 24) return {};
  /*
          if ( bmih.biCompression != BMP_BI_RGB ) {
                  return {};
          }
  */
  file.seekg(pos, std::fstream::beg);

  width = bmih.biWidth;
  height = bmih.biHeight;

  std::cout << "readBMP1 " << width << " " << height << '\n';

  int padWidth = width * 3;
  int pad = 0;
  if (padWidth % 4 != 0) {
    pad = 4 - (padWidth % 4);
    padWidth += pad;
  }
  int bytes = height * padWidth;

  std::vector<uint8_t> data(bytes);

  file.read(reinterpret_cast<char*>(data.data()), bytes);

  if (!file) {
    return {};
  }

  // shuffle bitmap data such that it is (R,G,B) tuples in row-major order
  int i = 0, j = 0;
  j = 0;
  uint8_t temp = 0;
  uint8_t* in = nullptr;
  uint8_t* out = nullptr;

  in = data.data();
  out = data.data();

  for (j = 0; j < height; ++j) {
    for (i = 0; i < width; ++i) {
      out[1] = in[1];
      temp = in[2];
      out[2] = in[0];
      out[0] = temp;

      in += 3;
      out += 3;
    }
    in += pad;
  }

  return data;
}

void writeBMP(const std::string& iname, int width, int height,
              const std::vector<uint8_t>& data) {
  int bytes = 0, pad = 0;
  bytes = width * 3;
  pad = (bytes % 4) ? 4 - (bytes % 4) : 0;
  bytes += pad;
  bytes *= height;

  bmfh.bfType = 0x4d42;  // "BM"
  bmfh.bfSize =
      sizeof(BMP_BITMAPFILEHEADER) + sizeof(BMP_BITMAPINFOHEADER) + bytes;
  bmfh.bfReserved1 = 0;
  bmfh.bfReserved2 = 0;
  bmfh.bfOffBits = /*hack sizeof(BMP_BITMAPFILEHEADER)=14, sizeof doesn't
                      work?*/
      14 + sizeof(BMP_BITMAPINFOHEADER);

  bmih.biSize = sizeof(BMP_BITMAPINFOHEADER);
  bmih.biWidth = width;
  bmih.biHeight = height;
  bmih.biPlanes = 1;
  bmih.biBitCount = 24;
  bmih.biCompression = BMP_BI_RGB;
  bmih.biSizeImage = 0;
  bmih.biXPelsPerMeter = (int)(100 / 2.54 * 72);
  bmih.biYPelsPerMeter = (int)(100 / 2.54 * 72);
  bmih.biClrUsed = 0;
  bmih.biClrImportant = 0;

  std::fstream outFile(iname, std::fstream::out | std::fstream::binary);

  //	outFile.write(reinterpret_cast<const char*>(&bmfh),
  // sizeof(BMP_BITMAPFILEHEADER));
  outFile.write(reinterpret_cast<const char*>(&(bmfh.bfType)), 2);
  outFile.write(reinterpret_cast<const char*>(&(bmfh.bfSize)), 4);
  outFile.write(reinterpret_cast<const char*>(&(bmfh.bfReserved1)), 2);
  outFile.write(reinterpret_cast<const char*>(&(bmfh.bfReserved2)), 2);
  outFile.write(reinterpret_cast<const char*>(&(bmfh.bfOffBits)), 4);

  outFile.write(reinterpret_cast<const char*>(&bmih),
                sizeof(BMP_BITMAPINFOHEADER));

  bytes /= height;
  std::vector<uint8_t> scanline(bytes);
  for (int j = 0; j < height; ++j) {
    memcpy(scanline.data(), &data[j * 3 * width], 3 * width);
    for (int i = 0; i < width; ++i) {
      uint8_t temp = scanline[i * 3];
      scanline[i * 3] = scanline[i * 3 + 2];
      scanline[i * 3 + 2] = temp;
    }
    outFile.write(reinterpret_cast<const char*>(scanline.data()), bytes);
  }
}