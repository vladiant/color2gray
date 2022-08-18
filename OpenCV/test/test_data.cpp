#include "test_data.hpp"

using namespace cv;

Mat get_test_image() {
  static const Mat image(
      {1, 172},
      {
          Vec3b{211, 147, 105}, Vec3b{211, 147, 106}, Vec3b{211, 147, 107},
          Vec3b{210, 147, 107}, Vec3b{210, 146, 108}, Vec3b{209, 146, 108},
          Vec3b{209, 146, 108}, Vec3b{208, 146, 109}, Vec3b{208, 146, 109},
          Vec3b{207, 145, 110}, Vec3b{207, 145, 110}, Vec3b{206, 145, 111},
          Vec3b{206, 144, 111}, Vec3b{205, 145, 112}, Vec3b{204, 144, 113},
          Vec3b{204, 144, 113}, Vec3b{204, 144, 114}, Vec3b{203, 143, 114},
          Vec3b{202, 143, 115}, Vec3b{202, 143, 116}, Vec3b{201, 143, 117},
          Vec3b{200, 142, 117}, Vec3b{200, 142, 118}, Vec3b{199, 142, 119},
          Vec3b{198, 141, 120}, Vec3b{198, 141, 120}, Vec3b{197, 141, 121},
          Vec3b{196, 141, 122}, Vec3b{196, 140, 122}, Vec3b{195, 140, 123},
          Vec3b{194, 139, 124}, Vec3b{193, 139, 124}, Vec3b{192, 139, 126},
          Vec3b{192, 139, 126}, Vec3b{191, 138, 127}, Vec3b{191, 138, 128},
          Vec3b{189, 138, 128}, Vec3b{188, 137, 130}, Vec3b{188, 137, 130},
          Vec3b{187, 137, 131}, Vec3b{186, 136, 132}, Vec3b{185, 136, 133},
          Vec3b{184, 136, 133}, Vec3b{184, 135, 135}, Vec3b{183, 134, 135},
          Vec3b{182, 135, 136}, Vec3b{181, 134, 137}, Vec3b{180, 134, 138},
          Vec3b{179, 133, 139}, Vec3b{179, 133, 140}, Vec3b{177, 132, 141},
          Vec3b{177, 132, 142}, Vec3b{176, 132, 143}, Vec3b{175, 131, 144},
          Vec3b{174, 131, 144}, Vec3b{173, 130, 145}, Vec3b{172, 130, 147},
          Vec3b{171, 130, 148}, Vec3b{170, 130, 148}, Vec3b{169, 129, 149},
          Vec3b{168, 128, 150}, Vec3b{167, 128, 151}, Vec3b{166, 128, 152},
          Vec3b{166, 127, 153}, Vec3b{164, 127, 154}, Vec3b{164, 127, 155},
          Vec3b{162, 126, 156}, Vec3b{162, 126, 157}, Vec3b{161, 125, 158},
          Vec3b{159, 125, 159}, Vec3b{159, 124, 161}, Vec3b{158, 124, 161},
          Vec3b{157, 123, 162}, Vec3b{156, 123, 163}, Vec3b{155, 123, 164},
          Vec3b{154, 122, 165}, Vec3b{153, 122, 166}, Vec3b{152, 121, 167},
          Vec3b{150, 121, 168}, Vec3b{150, 121, 170}, Vec3b{149, 120, 171},
          Vec3b{148, 120, 171}, Vec3b{147, 119, 173}, Vec3b{146, 118, 174},
          Vec3b{144, 118, 175}, Vec3b{144, 118, 175}, Vec3b{143, 117, 176},
          Vec3b{142, 117, 178}, Vec3b{141, 117, 179}, Vec3b{140, 116, 180},
          Vec3b{139, 116, 181}, Vec3b{138, 115, 182}, Vec3b{137, 115, 183},
          Vec3b{136, 114, 184}, Vec3b{135, 114, 185}, Vec3b{134, 114, 186},
          Vec3b{133, 113, 187}, Vec3b{131, 113, 188}, Vec3b{131, 113, 189},
          Vec3b{130, 112, 190}, Vec3b{129, 111, 191}, Vec3b{128, 111, 192},
          Vec3b{127, 111, 193}, Vec3b{126, 110, 194}, Vec3b{125, 110, 195},
          Vec3b{124, 109, 196}, Vec3b{123, 109, 197}, Vec3b{122, 108, 198},
          Vec3b{121, 108, 199}, Vec3b{120, 108, 200}, Vec3b{119, 107, 201},
          Vec3b{118, 107, 202}, Vec3b{117, 107, 203}, Vec3b{117, 106, 204},
          Vec3b{115, 106, 205}, Vec3b{115, 105, 206}, Vec3b{114, 105, 207},
          Vec3b{113, 104, 208}, Vec3b{112, 104, 208}, Vec3b{111, 103, 209},
          Vec3b{110, 103, 211}, Vec3b{109, 103, 212}, Vec3b{108, 102, 212},
          Vec3b{108, 102, 213}, Vec3b{107, 102, 214}, Vec3b{105, 101, 215},
          Vec3b{105, 101, 216}, Vec3b{104, 101, 217}, Vec3b{103, 100, 218},
          Vec3b{102, 100, 219}, Vec3b{101, 100, 220}, Vec3b{100, 99, 221},
          Vec3b{100, 99, 221},  Vec3b{99, 99, 222},   Vec3b{98, 98, 223},
          Vec3b{97, 98, 224},   Vec3b{96, 97, 225},   Vec3b{96, 97, 226},
          Vec3b{95, 97, 226},   Vec3b{94, 97, 227},   Vec3b{94, 96, 228},
          Vec3b{93, 96, 228},   Vec3b{92, 95, 229},   Vec3b{91, 95, 230},
          Vec3b{90, 95, 231},   Vec3b{90, 94, 231},   Vec3b{89, 94, 232},
          Vec3b{89, 94, 233},   Vec3b{88, 94, 234},   Vec3b{87, 94, 235},
          Vec3b{87, 93, 235},   Vec3b{86, 93, 236},   Vec3b{85, 92, 237},
          Vec3b{84, 92, 237},   Vec3b{84, 92, 238},   Vec3b{83, 92, 238},
          Vec3b{82, 92, 239},   Vec3b{82, 91, 240},   Vec3b{82, 91, 240},
          Vec3b{81, 91, 241},   Vec3b{80, 91, 242},   Vec3b{80, 91, 242},
          Vec3b{79, 90, 242},   Vec3b{78, 90, 243},   Vec3b{78, 90, 243},
          Vec3b{78, 89, 244},   Vec3b{77, 89, 245},   Vec3b{77, 89, 245},
          Vec3b{76, 89, 246},   Vec3b{76, 89, 246},   Vec3b{75, 88, 246},
          Vec3b{75, 88, 247},
      });
  return image;
}

namespace full_image {

cv::Mat get_expected_gray_image() {
  static const Mat image(
      {1, 172},
      {Vec3b{100, 100, 100}, Vec3b{100, 100, 100}, Vec3b{100, 100, 100},
       Vec3b{100, 100, 100}, Vec3b{101, 100, 100}, Vec3b{101, 100, 100},
       Vec3b{101, 100, 100}, Vec3b{101, 101, 101}, Vec3b{101, 101, 101},
       Vec3b{101, 101, 101}, Vec3b{101, 101, 101}, Vec3b{102, 101, 101},
       Vec3b{102, 101, 101}, Vec3b{102, 102, 102}, Vec3b{102, 102, 102},
       Vec3b{102, 102, 102}, Vec3b{103, 102, 102}, Vec3b{103, 102, 103},
       Vec3b{103, 103, 103}, Vec3b{103, 103, 103}, Vec3b{104, 103, 103},
       Vec3b{104, 104, 104}, Vec3b{104, 104, 104}, Vec3b{104, 104, 104},
       Vec3b{105, 105, 105}, Vec3b{105, 105, 105}, Vec3b{105, 105, 105},
       Vec3b{106, 105, 105}, Vec3b{106, 105, 105}, Vec3b{106, 106, 106},
       Vec3b{107, 106, 106}, Vec3b{107, 106, 106}, Vec3b{107, 107, 107},
       Vec3b{107, 107, 107}, Vec3b{108, 107, 108}, Vec3b{108, 108, 108},
       Vec3b{108, 108, 108}, Vec3b{109, 109, 109}, Vec3b{109, 109, 109},
       Vec3b{110, 109, 109}, Vec3b{110, 110, 110}, Vec3b{110, 110, 110},
       Vec3b{111, 110, 110}, Vec3b{111, 111, 111}, Vec3b{112, 111, 111},
       Vec3b{112, 111, 112}, Vec3b{112, 112, 112}, Vec3b{113, 112, 112},
       Vec3b{113, 113, 113}, Vec3b{114, 113, 113}, Vec3b{114, 114, 114},
       Vec3b{115, 114, 114}, Vec3b{115, 115, 115}, Vec3b{116, 115, 115},
       Vec3b{116, 115, 115}, Vec3b{116, 116, 116}, Vec3b{117, 117, 117},
       Vec3b{117, 117, 117}, Vec3b{118, 117, 117}, Vec3b{118, 118, 118},
       Vec3b{119, 118, 118}, Vec3b{119, 119, 119}, Vec3b{120, 119, 119},
       Vec3b{120, 120, 120}, Vec3b{121, 120, 120}, Vec3b{121, 120, 120},
       Vec3b{122, 121, 121}, Vec3b{122, 121, 121}, Vec3b{122, 122, 122},
       Vec3b{123, 123, 123}, Vec3b{124, 123, 123}, Vec3b{124, 123, 123},
       Vec3b{124, 124, 124}, Vec3b{125, 124, 124}, Vec3b{125, 125, 125},
       Vec3b{126, 125, 125}, Vec3b{126, 126, 126}, Vec3b{127, 126, 126},
       Vec3b{128, 127, 127}, Vec3b{128, 128, 128}, Vec3b{129, 128, 128},
       Vec3b{129, 128, 128}, Vec3b{130, 129, 129}, Vec3b{130, 130, 130},
       Vec3b{131, 130, 130}, Vec3b{131, 130, 130}, Vec3b{131, 131, 131},
       Vec3b{132, 132, 132}, Vec3b{133, 132, 132}, Vec3b{133, 133, 133},
       Vec3b{134, 133, 133}, Vec3b{134, 134, 134}, Vec3b{135, 134, 134},
       Vec3b{135, 135, 135}, Vec3b{136, 135, 135}, Vec3b{136, 136, 136},
       Vec3b{137, 136, 136}, Vec3b{138, 137, 137}, Vec3b{138, 137, 137},
       Vec3b{138, 138, 138}, Vec3b{139, 138, 138}, Vec3b{139, 139, 139},
       Vec3b{140, 139, 140}, Vec3b{140, 140, 140}, Vec3b{141, 140, 141},
       Vec3b{142, 141, 141}, Vec3b{142, 142, 142}, Vec3b{143, 142, 142},
       Vec3b{143, 143, 143}, Vec3b{144, 143, 143}, Vec3b{144, 144, 144},
       Vec3b{145, 144, 144}, Vec3b{145, 145, 145}, Vec3b{146, 145, 145},
       Vec3b{147, 146, 146}, Vec3b{147, 146, 147}, Vec3b{147, 147, 147},
       Vec3b{148, 148, 148}, Vec3b{148, 148, 148}, Vec3b{149, 149, 149},
       Vec3b{150, 149, 149}, Vec3b{150, 150, 150}, Vec3b{151, 150, 150},
       Vec3b{151, 151, 151}, Vec3b{152, 151, 151}, Vec3b{153, 152, 152},
       Vec3b{153, 152, 152}, Vec3b{154, 153, 153}, Vec3b{154, 154, 154},
       Vec3b{155, 154, 154}, Vec3b{155, 155, 155}, Vec3b{156, 156, 156},
       Vec3b{156, 156, 156}, Vec3b{157, 156, 156}, Vec3b{157, 157, 157},
       Vec3b{158, 157, 158}, Vec3b{159, 158, 158}, Vec3b{159, 158, 158},
       Vec3b{159, 159, 159}, Vec3b{160, 159, 159}, Vec3b{160, 160, 160},
       Vec3b{161, 160, 160}, Vec3b{161, 161, 161}, Vec3b{162, 161, 161},
       Vec3b{163, 162, 162}, Vec3b{163, 162, 162}, Vec3b{163, 163, 163},
       Vec3b{163, 163, 163}, Vec3b{164, 164, 164}, Vec3b{165, 164, 164},
       Vec3b{165, 164, 164}, Vec3b{165, 165, 165}, Vec3b{166, 165, 166},
       Vec3b{166, 166, 166}, Vec3b{167, 166, 166}, Vec3b{167, 166, 167},
       Vec3b{168, 167, 167}, Vec3b{168, 167, 167}, Vec3b{168, 167, 167},
       Vec3b{169, 168, 168}, Vec3b{169, 168, 169}, Vec3b{169, 168, 169},
       Vec3b{170, 169, 169}, Vec3b{170, 169, 170}, Vec3b{170, 169, 170},
       Vec3b{170, 170, 170}, Vec3b{171, 170, 170}, Vec3b{171, 170, 170},
       Vec3b{171, 171, 171}, Vec3b{171, 171, 171}, Vec3b{172, 171, 171},
       Vec3b{172, 171, 171}});
  return image;
}

cv::Mat get_expected_color_image() {
  static const Mat image(
      {1, 172},
      {Vec3b{154, 103, 69},  Vec3b{154, 103, 70},  Vec3b{154, 103, 71},
       Vec3b{154, 103, 71},  Vec3b{155, 103, 73},  Vec3b{154, 103, 73},
       Vec3b{154, 103, 73},  Vec3b{154, 103, 74},  Vec3b{154, 103, 74},
       Vec3b{154, 104, 75},  Vec3b{154, 104, 75},  Vec3b{153, 104, 76},
       Vec3b{154, 104, 77},  Vec3b{153, 104, 77},  Vec3b{153, 104, 79},
       Vec3b{153, 104, 79},  Vec3b{153, 104, 79},  Vec3b{153, 104, 80},
       Vec3b{153, 104, 81},  Vec3b{153, 104, 82},  Vec3b{152, 105, 83},
       Vec3b{152, 105, 84},  Vec3b{153, 105, 85},  Vec3b{152, 105, 86},
       Vec3b{152, 105, 87},  Vec3b{152, 105, 87},  Vec3b{152, 105, 88},
       Vec3b{151, 105, 89},  Vec3b{152, 105, 90},  Vec3b{152, 106, 91},
       Vec3b{152, 106, 93},  Vec3b{151, 106, 93},  Vec3b{151, 106, 95},
       Vec3b{151, 106, 95},  Vec3b{151, 106, 97},  Vec3b{151, 106, 98},
       Vec3b{150, 107, 98},  Vec3b{151, 107, 101}, Vec3b{151, 107, 101},
       Vec3b{150, 107, 102}, Vec3b{151, 107, 104}, Vec3b{150, 108, 105},
       Vec3b{149, 108, 105}, Vec3b{151, 108, 108}, Vec3b{151, 108, 109},
       Vec3b{149, 108, 109}, Vec3b{150, 109, 111}, Vec3b{149, 109, 112},
       Vec3b{150, 109, 114}, Vec3b{150, 109, 115}, Vec3b{149, 109, 117},
       Vec3b{150, 110, 118}, Vec3b{149, 110, 120}, Vec3b{149, 110, 121},
       Vec3b{149, 110, 122}, Vec3b{149, 110, 124}, Vec3b{149, 111, 126},
       Vec3b{148, 111, 127}, Vec3b{148, 111, 127}, Vec3b{148, 111, 129},
       Vec3b{148, 111, 132}, Vec3b{148, 112, 133}, Vec3b{147, 112, 134},
       Vec3b{148, 112, 136}, Vec3b{147, 112, 137}, Vec3b{147, 113, 138},
       Vec3b{146, 113, 141}, Vec3b{147, 113, 142}, Vec3b{147, 113, 144},
       Vec3b{146, 114, 146}, Vec3b{147, 113, 149}, Vec3b{146, 114, 149},
       Vec3b{146, 114, 151}, Vec3b{146, 114, 152}, Vec3b{145, 114, 154},
       Vec3b{145, 114, 156}, Vec3b{145, 115, 157}, Vec3b{145, 115, 159},
       Vec3b{144, 115, 161}, Vec3b{144, 115, 163}, Vec3b{144, 116, 165},
       Vec3b{143, 116, 166}, Vec3b{144, 116, 169}, Vec3b{144, 116, 172},
       Vec3b{142, 117, 173}, Vec3b{142, 117, 173}, Vec3b{143, 117, 176},
       Vec3b{142, 117, 178}, Vec3b{141, 117, 179}, Vec3b{142, 117, 182},
       Vec3b{141, 118, 183}, Vec3b{141, 118, 186}, Vec3b{140, 118, 187},
       Vec3b{140, 118, 190}, Vec3b{140, 118, 191}, Vec3b{139, 119, 193},
       Vec3b{139, 119, 195}, Vec3b{138, 119, 197}, Vec3b{138, 119, 198},
       Vec3b{138, 119, 201}, Vec3b{138, 119, 203}, Vec3b{137, 120, 205},
       Vec3b{137, 120, 206}, Vec3b{137, 120, 209}, Vec3b{136, 120, 210},
       Vec3b{136, 120, 213}, Vec3b{136, 121, 215}, Vec3b{136, 121, 217},
       Vec3b{135, 121, 219}, Vec3b{134, 121, 220}, Vec3b{134, 121, 223},
       Vec3b{134, 122, 225}, Vec3b{133, 122, 226}, Vec3b{134, 122, 229},
       Vec3b{132, 122, 231}, Vec3b{133, 122, 233}, Vec3b{132, 122, 235},
       Vec3b{132, 122, 238}, Vec3b{132, 123, 238}, Vec3b{132, 123, 241},
       Vec3b{131, 123, 244}, Vec3b{130, 124, 246}, Vec3b{130, 124, 247},
       Vec3b{130, 124, 249}, Vec3b{129, 124, 250}, Vec3b{129, 124, 254},
       Vec3b{129, 124, 255}, Vec3b{128, 125, 255}, Vec3b{128, 125, 255},
       Vec3b{127, 125, 255}, Vec3b{126, 126, 255}, Vec3b{126, 126, 255},
       Vec3b{126, 126, 255}, Vec3b{126, 126, 255}, Vec3b{126, 126, 255},
       Vec3b{125, 126, 255}, Vec3b{125, 126, 255}, Vec3b{125, 126, 255},
       Vec3b{124, 127, 255}, Vec3b{123, 127, 255}, Vec3b{124, 127, 255},
       Vec3b{123, 127, 255}, Vec3b{123, 127, 255}, Vec3b{122, 128, 255},
       Vec3b{121, 128, 255}, Vec3b{122, 128, 255}, Vec3b{121, 128, 255},
       Vec3b{121, 128, 255}, Vec3b{120, 128, 255}, Vec3b{120, 129, 255},
       Vec3b{120, 128, 255}, Vec3b{119, 129, 255}, Vec3b{119, 129, 255},
       Vec3b{118, 129, 255}, Vec3b{118, 129, 255}, Vec3b{118, 129, 255},
       Vec3b{117, 130, 255}, Vec3b{117, 129, 255}, Vec3b{117, 129, 255},
       Vec3b{116, 130, 255}, Vec3b{115, 130, 255}, Vec3b{115, 130, 255},
       Vec3b{115, 130, 255}, Vec3b{114, 130, 255}, Vec3b{114, 130, 255},
       Vec3b{115, 130, 255}, Vec3b{114, 130, 255}, Vec3b{114, 130, 255},
       Vec3b{113, 130, 255}, Vec3b{113, 130, 255}, Vec3b{113, 130, 255},
       Vec3b{112, 130, 255}});
  return image;
}

}  // namespace full_image

namespace mu_image {

cv::Mat get_expected_gray_image() {
  static const Mat image(
      {1, 172},
      {Vec3b{76, 76, 76},    Vec3b{77, 77, 77},    Vec3b{78, 78, 78},
       Vec3b{78, 78, 78},    Vec3b{80, 79, 80},    Vec3b{80, 80, 80},
       Vec3b{80, 80, 80},    Vec3b{81, 81, 81},    Vec3b{81, 81, 81},
       Vec3b{83, 83, 83},    Vec3b{83, 83, 83},    Vec3b{84, 84, 84},
       Vec3b{84, 84, 84},    Vec3b{85, 85, 85},    Vec3b{87, 87, 87},
       Vec3b{87, 86, 86},    Vec3b{87, 87, 87},    Vec3b{88, 88, 88},
       Vec3b{89, 89, 89},    Vec3b{90, 89, 90},    Vec3b{91, 91, 91},
       Vec3b{91, 91, 91},    Vec3b{92, 92, 92},    Vec3b{93, 93, 93},
       Vec3b{95, 94, 94},    Vec3b{94, 93, 93},    Vec3b{95, 94, 94},
       Vec3b{96, 95, 95},    Vec3b{95, 95, 95},    Vec3b{96, 96, 96},
       Vec3b{98, 97, 97},    Vec3b{97, 97, 97},    Vec3b{99, 99, 99},
       Vec3b{98, 98, 98},    Vec3b{100, 99, 99},   Vec3b{100, 99, 99},
       Vec3b{100, 100, 100}, Vec3b{103, 102, 102}, Vec3b{101, 101, 101},
       Vec3b{102, 102, 102}, Vec3b{103, 103, 103}, Vec3b{104, 103, 103},
       Vec3b{103, 103, 103}, Vec3b{105, 105, 105}, Vec3b{105, 104, 104},
       Vec3b{105, 104, 105}, Vec3b{106, 105, 106}, Vec3b{106, 106, 106},
       Vec3b{107, 107, 107}, Vec3b{107, 107, 107}, Vec3b{109, 109, 109},
       Vec3b{109, 108, 108}, Vec3b{109, 109, 109}, Vec3b{110, 110, 110},
       Vec3b{109, 109, 109}, Vec3b{110, 109, 109}, Vec3b{112, 111, 111},
       Vec3b{112, 112, 112}, Vec3b{111, 111, 111}, Vec3b{112, 111, 111},
       Vec3b{113, 112, 112}, Vec3b{113, 113, 113}, Vec3b{113, 113, 113},
       Vec3b{113, 113, 113}, Vec3b{114, 114, 114}, Vec3b{114, 113, 113},
       Vec3b{115, 115, 115}, Vec3b{115, 114, 114}, Vec3b{115, 115, 115},
       Vec3b{116, 116, 116}, Vec3b{118, 117, 118}, Vec3b{117, 116, 116},
       Vec3b{117, 117, 117}, Vec3b{117, 117, 117}, Vec3b{118, 117, 117},
       Vec3b{118, 118, 118}, Vec3b{118, 118, 118}, Vec3b{119, 119, 119},
       Vec3b{121, 120, 120}, Vec3b{121, 121, 121}, Vec3b{122, 121, 121},
       Vec3b{120, 120, 120}, Vec3b{123, 122, 122}, Vec3b{123, 123, 123},
       Vec3b{125, 124, 124}, Vec3b{122, 122, 122}, Vec3b{123, 122, 123},
       Vec3b{125, 124, 124}, Vec3b{125, 124, 124}, Vec3b{125, 125, 125},
       Vec3b{126, 125, 125}, Vec3b{126, 126, 126}, Vec3b{127, 126, 126},
       Vec3b{127, 127, 127}, Vec3b{128, 127, 127}, Vec3b{128, 128, 128},
       Vec3b{129, 128, 129}, Vec3b{131, 130, 130}, Vec3b{130, 129, 129},
       Vec3b{131, 130, 130}, Vec3b{132, 131, 131}, Vec3b{132, 131, 132},
       Vec3b{132, 132, 132}, Vec3b{133, 133, 133}, Vec3b{134, 133, 134},
       Vec3b{135, 134, 135}, Vec3b{136, 135, 135}, Vec3b{137, 136, 136},
       Vec3b{137, 137, 137}, Vec3b{138, 138, 138}, Vec3b{139, 139, 139},
       Vec3b{140, 139, 140}, Vec3b{141, 140, 140}, Vec3b{140, 140, 140},
       Vec3b{143, 142, 142}, Vec3b{143, 142, 142}, Vec3b{144, 143, 143},
       Vec3b{145, 145, 145}, Vec3b{145, 144, 144}, Vec3b{146, 146, 146},
       Vec3b{149, 148, 149}, Vec3b{150, 150, 150}, Vec3b{150, 150, 150},
       Vec3b{150, 149, 149}, Vec3b{151, 151, 151}, Vec3b{155, 155, 155},
       Vec3b{155, 154, 154}, Vec3b{156, 156, 156}, Vec3b{158, 158, 158},
       Vec3b{160, 160, 160}, Vec3b{162, 161, 162}, Vec3b{164, 164, 164},
       Vec3b{162, 162, 162}, Vec3b{164, 164, 164}, Vec3b{167, 166, 166},
       Vec3b{169, 169, 169}, Vec3b{172, 171, 171}, Vec3b{172, 171, 171},
       Vec3b{173, 172, 172}, Vec3b{175, 175, 175}, Vec3b{176, 175, 175},
       Vec3b{177, 176, 176}, Vec3b{180, 180, 180}, Vec3b{183, 183, 183},
       Vec3b{186, 186, 186}, Vec3b{186, 185, 185}, Vec3b{189, 188, 188},
       Vec3b{189, 188, 189}, Vec3b{193, 192, 192}, Vec3b{196, 196, 196},
       Vec3b{196, 195, 195}, Vec3b{200, 199, 199}, Vec3b{204, 203, 203},
       Vec3b{206, 206, 206}, Vec3b{207, 206, 207}, Vec3b{210, 209, 210},
       Vec3b{215, 214, 214}, Vec3b{216, 215, 216}, Vec3b{216, 215, 215},
       Vec3b{221, 220, 220}, Vec3b{226, 225, 225}, Vec3b{225, 225, 225},
       Vec3b{229, 228, 229}, Vec3b{235, 234, 234}, Vec3b{235, 234, 234},
       Vec3b{238, 237, 237}, Vec3b{243, 242, 242}, Vec3b{243, 242, 242},
       Vec3b{249, 248, 249}, Vec3b{249, 248, 249}, Vec3b{254, 253, 253},
       Vec3b{255, 255, 255}});
  return image;
}

cv::Mat get_expected_color_image() {
  static const Mat3b image(
      {1, 172},
      {Vec3b{122, 79, 50},   Vec3b{123, 80, 52},   Vec3b{124, 80, 53},
       Vec3b{124, 81, 53},   Vec3b{127, 82, 55},   Vec3b{127, 82, 56},
       Vec3b{126, 82, 56},   Vec3b{127, 84, 57},   Vec3b{127, 83, 57},
       Vec3b{130, 85, 60},   Vec3b{129, 85, 60},   Vec3b{130, 86, 61},
       Vec3b{131, 86, 62},   Vec3b{131, 87, 63},   Vec3b{133, 89, 66},
       Vec3b{132, 88, 65},   Vec3b{133, 89, 66},   Vec3b{134, 89, 67},
       Vec3b{134, 90, 69},   Vec3b{135, 91, 70},   Vec3b{136, 92, 72},
       Vec3b{136, 92, 73},   Vec3b{137, 92, 74},   Vec3b{137, 93, 75},
       Vec3b{139, 95, 78},   Vec3b{138, 94, 77},   Vec3b{138, 95, 79},
       Vec3b{139, 95, 80},   Vec3b{139, 95, 81},   Vec3b{139, 96, 82},
       Vec3b{141, 97, 85},   Vec3b{139, 96, 84},   Vec3b{141, 98, 88},
       Vec3b{140, 97, 87},   Vec3b{141, 98, 89},   Vec3b{141, 98, 90},
       Vec3b{140, 99, 90},   Vec3b{143, 101, 95},  Vec3b{141, 99, 93},
       Vec3b{141, 100, 95},  Vec3b{142, 101, 97},  Vec3b{142, 101, 98},
       Vec3b{140, 100, 98},  Vec3b{143, 102, 102}, Vec3b{142, 101, 102},
       Vec3b{141, 102, 102}, Vec3b{142, 102, 105}, Vec3b{141, 103, 106},
       Vec3b{143, 103, 108}, Vec3b{142, 103, 109}, Vec3b{143, 104, 112},
       Vec3b{142, 104, 112}, Vec3b{142, 104, 113}, Vec3b{143, 104, 116},
       Vec3b{141, 104, 115}, Vec3b{141, 104, 117}, Vec3b{142, 105, 120},
       Vec3b{142, 106, 121}, Vec3b{140, 105, 121}, Vec3b{140, 105, 123},
       Vec3b{141, 106, 125}, Vec3b{140, 106, 126}, Vec3b{140, 106, 127},
       Vec3b{140, 105, 129}, Vec3b{140, 107, 131}, Vec3b{139, 106, 131},
       Vec3b{139, 107, 134}, Vec3b{138, 106, 134}, Vec3b{139, 106, 136},
       Vec3b{138, 107, 138}, Vec3b{140, 108, 142}, Vec3b{138, 107, 141},
       Vec3b{138, 107, 143}, Vec3b{137, 107, 144}, Vec3b{137, 107, 145},
       Vec3b{137, 107, 147}, Vec3b{136, 107, 148}, Vec3b{136, 108, 150},
       Vec3b{136, 109, 153}, Vec3b{136, 109, 155}, Vec3b{136, 109, 157},
       Vec3b{134, 108, 156}, Vec3b{136, 109, 161}, Vec3b{136, 110, 163},
       Vec3b{136, 111, 166}, Vec3b{133, 109, 163}, Vec3b{134, 109, 165},
       Vec3b{134, 110, 169}, Vec3b{133, 110, 170}, Vec3b{134, 110, 172},
       Vec3b{133, 110, 174}, Vec3b{133, 111, 176}, Vec3b{132, 111, 177},
       Vec3b{133, 111, 180}, Vec3b{132, 111, 181}, Vec3b{131, 111, 182},
       Vec3b{131, 111, 185}, Vec3b{131, 113, 188}, Vec3b{130, 112, 188},
       Vec3b{130, 112, 191}, Vec3b{131, 113, 194}, Vec3b{130, 113, 195},
       Vec3b{129, 113, 196}, Vec3b{130, 114, 199}, Vec3b{129, 114, 201},
       Vec3b{130, 114, 204}, Vec3b{129, 115, 206}, Vec3b{130, 115, 209},
       Vec3b{129, 116, 211}, Vec3b{129, 116, 213}, Vec3b{129, 117, 216},
       Vec3b{129, 117, 218}, Vec3b{128, 118, 220}, Vec3b{129, 117, 221},
       Vec3b{129, 119, 226}, Vec3b{129, 118, 227}, Vec3b{129, 119, 229},
       Vec3b{130, 120, 233}, Vec3b{128, 120, 233}, Vec3b{129, 120, 237},
       Vec3b{130, 122, 243}, Vec3b{130, 123, 245}, Vec3b{130, 123, 247},
       Vec3b{129, 122, 247}, Vec3b{129, 124, 250}, Vec3b{131, 127, 255},
       Vec3b{130, 126, 255}, Vec3b{130, 127, 255}, Vec3b{132, 128, 255},
       Vec3b{132, 130, 255}, Vec3b{132, 131, 255}, Vec3b{134, 133, 255},
       Vec3b{132, 131, 255}, Vec3b{132, 133, 255}, Vec3b{134, 134, 255},
       Vec3b{134, 136, 255}, Vec3b{136, 138, 255}, Vec3b{136, 138, 255},
       Vec3b{135, 139, 255}, Vec3b{136, 141, 255}, Vec3b{137, 140, 255},
       Vec3b{137, 141, 255}, Vec3b{139, 144, 255}, Vec3b{140, 146, 255},
       Vec3b{141, 148, 255}, Vec3b{141, 147, 255}, Vec3b{142, 150, 255},
       Vec3b{142, 150, 255}, Vec3b{144, 153, 255}, Vec3b{145, 156, 255},
       Vec3b{146, 155, 255}, Vec3b{147, 158, 255}, Vec3b{150, 161, 255},
       Vec3b{151, 163, 255}, Vec3b{151, 164, 255}, Vec3b{152, 166, 255},
       Vec3b{154, 170, 255}, Vec3b{156, 171, 255}, Vec3b{155, 170, 255},
       Vec3b{158, 174, 255}, Vec3b{160, 179, 255}, Vec3b{160, 178, 255},
       Vec3b{162, 181, 255}, Vec3b{165, 186, 255}, Vec3b{164, 185, 255},
       Vec3b{168, 187, 255}, Vec3b{170, 192, 255}, Vec3b{170, 192, 255},
       Vec3b{173, 197, 255}, Vec3b{173, 197, 255}, Vec3b{176, 201, 255},
       Vec3b{178, 202, 255}});
  return image;
}

}  // namespace mu_image

namespace quantized_image {

Mat get_expected_quantized_image() {
  static const Mat3b image(
      {1, 172},
      {Vec3b{207, 145, 110}, Vec3b{207, 145, 110}, Vec3b{207, 145, 110},
       Vec3b{207, 145, 110}, Vec3b{207, 145, 110}, Vec3b{207, 145, 110},
       Vec3b{207, 145, 110}, Vec3b{207, 145, 110}, Vec3b{207, 145, 110},
       Vec3b{207, 145, 110}, Vec3b{207, 145, 110}, Vec3b{207, 145, 110},
       Vec3b{207, 145, 110}, Vec3b{207, 145, 110}, Vec3b{207, 145, 110},
       Vec3b{207, 145, 110}, Vec3b{207, 145, 110}, Vec3b{207, 145, 110},
       Vec3b{199, 142, 119}, Vec3b{199, 142, 119}, Vec3b{199, 142, 119},
       Vec3b{199, 142, 119}, Vec3b{199, 142, 119}, Vec3b{199, 142, 119},
       Vec3b{199, 142, 119}, Vec3b{199, 142, 119}, Vec3b{199, 142, 119},
       Vec3b{199, 142, 119}, Vec3b{199, 142, 119}, Vec3b{199, 142, 119},
       Vec3b{191, 138, 127}, Vec3b{191, 138, 127}, Vec3b{191, 138, 127},
       Vec3b{191, 138, 127}, Vec3b{191, 138, 127}, Vec3b{191, 138, 127},
       Vec3b{191, 138, 127}, Vec3b{191, 138, 127}, Vec3b{191, 138, 127},
       Vec3b{184, 135, 134}, Vec3b{184, 135, 134}, Vec3b{184, 135, 134},
       Vec3b{184, 135, 134}, Vec3b{184, 135, 134}, Vec3b{184, 135, 134},
       Vec3b{184, 135, 134}, Vec3b{184, 135, 134}, Vec3b{177, 132, 141},
       Vec3b{177, 132, 141}, Vec3b{177, 132, 141}, Vec3b{177, 132, 141},
       Vec3b{177, 132, 141}, Vec3b{177, 132, 141}, Vec3b{177, 132, 141},
       Vec3b{177, 132, 141}, Vec3b{169, 129, 149}, Vec3b{169, 129, 149},
       Vec3b{169, 129, 149}, Vec3b{169, 129, 149}, Vec3b{169, 129, 149},
       Vec3b{169, 129, 149}, Vec3b{169, 129, 149}, Vec3b{169, 129, 149},
       Vec3b{169, 129, 149}, Vec3b{162, 126, 156}, Vec3b{162, 126, 156},
       Vec3b{162, 126, 156}, Vec3b{162, 126, 156}, Vec3b{162, 126, 156},
       Vec3b{162, 126, 156}, Vec3b{156, 123, 163}, Vec3b{156, 123, 163},
       Vec3b{156, 123, 163}, Vec3b{156, 123, 163}, Vec3b{156, 123, 163},
       Vec3b{156, 123, 163}, Vec3b{150, 121, 169}, Vec3b{150, 121, 169},
       Vec3b{150, 121, 169}, Vec3b{150, 121, 169}, Vec3b{150, 121, 169},
       Vec3b{150, 121, 169}, Vec3b{145, 118, 175}, Vec3b{145, 118, 175},
       Vec3b{145, 118, 175}, Vec3b{145, 118, 175}, Vec3b{145, 118, 175},
       Vec3b{138, 116, 182}, Vec3b{138, 116, 182}, Vec3b{138, 116, 182},
       Vec3b{138, 116, 182}, Vec3b{138, 116, 182}, Vec3b{138, 116, 182},
       Vec3b{138, 116, 182}, Vec3b{138, 116, 182}, Vec3b{129, 112, 190},
       Vec3b{129, 112, 190}, Vec3b{129, 112, 190}, Vec3b{129, 112, 190},
       Vec3b{129, 112, 190}, Vec3b{129, 112, 190}, Vec3b{129, 112, 190},
       Vec3b{129, 112, 190}, Vec3b{129, 112, 190}, Vec3b{129, 112, 190},
       Vec3b{119, 107, 201}, Vec3b{119, 107, 201}, Vec3b{119, 107, 201},
       Vec3b{119, 107, 201}, Vec3b{119, 107, 201}, Vec3b{119, 107, 201},
       Vec3b{119, 107, 201}, Vec3b{119, 107, 201}, Vec3b{119, 107, 201},
       Vec3b{119, 107, 201}, Vec3b{119, 107, 201}, Vec3b{108, 102, 213},
       Vec3b{108, 102, 213}, Vec3b{108, 102, 213}, Vec3b{108, 102, 213},
       Vec3b{108, 102, 213}, Vec3b{108, 102, 213}, Vec3b{108, 102, 213},
       Vec3b{108, 102, 213}, Vec3b{108, 102, 213}, Vec3b{108, 102, 213},
       Vec3b{108, 102, 213}, Vec3b{108, 102, 213}, Vec3b{108, 102, 213},
       Vec3b{108, 102, 213}, Vec3b{94, 97, 227},   Vec3b{94, 97, 227},
       Vec3b{94, 97, 227},   Vec3b{94, 97, 227},   Vec3b{94, 97, 227},
       Vec3b{94, 97, 227},   Vec3b{94, 97, 227},   Vec3b{94, 97, 227},
       Vec3b{94, 97, 227},   Vec3b{94, 97, 227},   Vec3b{94, 97, 227},
       Vec3b{94, 97, 227},   Vec3b{94, 97, 227},   Vec3b{94, 97, 227},
       Vec3b{94, 97, 227},   Vec3b{94, 97, 227},   Vec3b{94, 97, 227},
       Vec3b{94, 97, 227},   Vec3b{94, 97, 227},   Vec3b{81, 91, 241},
       Vec3b{81, 91, 241},   Vec3b{81, 91, 241},   Vec3b{81, 91, 241},
       Vec3b{81, 91, 241},   Vec3b{81, 91, 241},   Vec3b{81, 91, 241},
       Vec3b{81, 91, 241},   Vec3b{81, 91, 241},   Vec3b{81, 91, 241},
       Vec3b{81, 91, 241},   Vec3b{81, 91, 241},   Vec3b{81, 91, 241},
       Vec3b{81, 91, 241},   Vec3b{81, 91, 241},   Vec3b{81, 91, 241},
       Vec3b{81, 91, 241},   Vec3b{81, 91, 241},   Vec3b{81, 91, 241},
       Vec3b{81, 91, 241},   Vec3b{81, 91, 241},   Vec3b{81, 91, 241},
       Vec3b{81, 91, 241}});
  return image;
}

Mat get_expected_gray_image() {
  static const Mat3b image(
      {1, 172},
      {Vec3b{76, 76, 76},    Vec3b{77, 77, 77},    Vec3b{78, 78, 78},
       Vec3b{78, 78, 78},    Vec3b{80, 79, 80},    Vec3b{80, 80, 80},
       Vec3b{80, 80, 80},    Vec3b{81, 81, 81},    Vec3b{81, 81, 81},
       Vec3b{83, 83, 83},    Vec3b{83, 83, 83},    Vec3b{84, 84, 84},
       Vec3b{85, 84, 84},    Vec3b{85, 85, 85},    Vec3b{87, 87, 87},
       Vec3b{87, 86, 86},    Vec3b{87, 87, 87},    Vec3b{88, 88, 88},
       Vec3b{89, 89, 89},    Vec3b{90, 90, 90},    Vec3b{91, 91, 91},
       Vec3b{91, 91, 91},    Vec3b{92, 92, 92},    Vec3b{93, 93, 93},
       Vec3b{95, 94, 94},    Vec3b{94, 93, 93},    Vec3b{95, 94, 94},
       Vec3b{96, 95, 95},    Vec3b{95, 95, 95},    Vec3b{96, 96, 96},
       Vec3b{98, 97, 97},    Vec3b{97, 97, 97},    Vec3b{100, 99, 99},
       Vec3b{98, 98, 98},    Vec3b{100, 99, 99},   Vec3b{100, 99, 99},
       Vec3b{100, 100, 100}, Vec3b{103, 102, 102}, Vec3b{101, 101, 101},
       Vec3b{102, 102, 102}, Vec3b{103, 103, 103}, Vec3b{104, 103, 103},
       Vec3b{103, 103, 103}, Vec3b{105, 105, 105}, Vec3b{105, 104, 104},
       Vec3b{105, 104, 105}, Vec3b{106, 106, 106}, Vec3b{106, 106, 106},
       Vec3b{107, 107, 107}, Vec3b{107, 107, 107}, Vec3b{109, 109, 109},
       Vec3b{109, 108, 108}, Vec3b{109, 109, 109}, Vec3b{110, 110, 110},
       Vec3b{109, 109, 109}, Vec3b{110, 109, 110}, Vec3b{112, 111, 111},
       Vec3b{112, 112, 112}, Vec3b{111, 111, 111}, Vec3b{112, 111, 112},
       Vec3b{113, 112, 112}, Vec3b{113, 113, 113}, Vec3b{113, 113, 113},
       Vec3b{113, 113, 113}, Vec3b{114, 114, 114}, Vec3b{114, 113, 113},
       Vec3b{115, 115, 115}, Vec3b{115, 114, 114}, Vec3b{115, 115, 115},
       Vec3b{116, 116, 116}, Vec3b{118, 117, 118}, Vec3b{117, 116, 116},
       Vec3b{117, 117, 117}, Vec3b{117, 117, 117}, Vec3b{118, 117, 117},
       Vec3b{118, 118, 118}, Vec3b{119, 118, 118}, Vec3b{119, 119, 119},
       Vec3b{121, 120, 120}, Vec3b{121, 121, 121}, Vec3b{122, 121, 121},
       Vec3b{120, 120, 120}, Vec3b{123, 122, 122}, Vec3b{123, 123, 123},
       Vec3b{125, 124, 124}, Vec3b{122, 122, 122}, Vec3b{123, 123, 123},
       Vec3b{125, 124, 124}, Vec3b{125, 124, 124}, Vec3b{126, 125, 125},
       Vec3b{126, 125, 125}, Vec3b{127, 126, 126}, Vec3b{127, 126, 126},
       Vec3b{127, 127, 127}, Vec3b{128, 127, 127}, Vec3b{128, 128, 128},
       Vec3b{129, 128, 129}, Vec3b{131, 130, 130}, Vec3b{130, 129, 129},
       Vec3b{131, 130, 130}, Vec3b{132, 131, 131}, Vec3b{132, 132, 132},
       Vec3b{132, 132, 132}, Vec3b{133, 133, 133}, Vec3b{134, 133, 134},
       Vec3b{135, 134, 135}, Vec3b{136, 135, 135}, Vec3b{137, 136, 136},
       Vec3b{137, 137, 137}, Vec3b{138, 138, 138}, Vec3b{139, 139, 139},
       Vec3b{140, 139, 140}, Vec3b{141, 140, 140}, Vec3b{140, 140, 140},
       Vec3b{143, 142, 142}, Vec3b{143, 142, 142}, Vec3b{144, 143, 143},
       Vec3b{145, 145, 145}, Vec3b{145, 144, 144}, Vec3b{146, 146, 146},
       Vec3b{149, 148, 149}, Vec3b{150, 150, 150}, Vec3b{150, 150, 150},
       Vec3b{150, 149, 149}, Vec3b{151, 151, 151}, Vec3b{155, 155, 155},
       Vec3b{155, 154, 154}, Vec3b{156, 156, 156}, Vec3b{158, 158, 158},
       Vec3b{160, 160, 160}, Vec3b{162, 161, 162}, Vec3b{164, 164, 164},
       Vec3b{162, 162, 162}, Vec3b{164, 164, 164}, Vec3b{167, 166, 167},
       Vec3b{169, 169, 169}, Vec3b{172, 171, 172}, Vec3b{172, 171, 171},
       Vec3b{173, 172, 172}, Vec3b{175, 175, 175}, Vec3b{176, 175, 175},
       Vec3b{177, 176, 177}, Vec3b{180, 180, 180}, Vec3b{183, 183, 183},
       Vec3b{186, 186, 186}, Vec3b{186, 185, 185}, Vec3b{189, 188, 188},
       Vec3b{189, 188, 189}, Vec3b{193, 192, 192}, Vec3b{196, 196, 196},
       Vec3b{196, 195, 195}, Vec3b{200, 199, 199}, Vec3b{204, 203, 203},
       Vec3b{206, 206, 206}, Vec3b{207, 206, 207}, Vec3b{210, 209, 210},
       Vec3b{215, 214, 214}, Vec3b{216, 215, 216}, Vec3b{216, 215, 215},
       Vec3b{221, 220, 220}, Vec3b{226, 225, 225}, Vec3b{225, 225, 225},
       Vec3b{229, 228, 229}, Vec3b{235, 234, 234}, Vec3b{235, 234, 234},
       Vec3b{238, 237, 237}, Vec3b{243, 242, 242}, Vec3b{243, 242, 242},
       Vec3b{249, 248, 249}, Vec3b{249, 248, 249}, Vec3b{254, 253, 253},
       Vec3b{255, 255, 255}});
  return image;
}

Mat get_expected_color_image() {
  static const Mat3b image(
      {1, 172},
      {Vec3b{120, 78, 54},   Vec3b{121, 79, 55},   Vec3b{123, 80, 56},
       Vec3b{123, 80, 56},   Vec3b{125, 82, 57},   Vec3b{126, 82, 58},
       Vec3b{125, 82, 57},   Vec3b{127, 83, 59},   Vec3b{127, 83, 58},
       Vec3b{130, 85, 60},   Vec3b{129, 85, 60},   Vec3b{131, 86, 61},
       Vec3b{131, 86, 61},   Vec3b{133, 87, 62},   Vec3b{135, 89, 63},
       Vec3b{134, 89, 63},   Vec3b{135, 89, 64},   Vec3b{136, 90, 64},
       Vec3b{132, 90, 72},   Vec3b{133, 90, 73},   Vec3b{135, 91, 74},
       Vec3b{135, 92, 74},   Vec3b{136, 92, 75},   Vec3b{137, 93, 76},
       Vec3b{139, 95, 77},   Vec3b{138, 94, 76},   Vec3b{139, 95, 77},
       Vec3b{141, 96, 78},   Vec3b{140, 96, 78},   Vec3b{141, 97, 78},
       Vec3b{139, 96, 87},   Vec3b{138, 96, 87},   Vec3b{141, 98, 89},
       Vec3b{139, 97, 88},   Vec3b{141, 98, 89},   Vec3b{141, 98, 89},
       Vec3b{142, 99, 90},   Vec3b{145, 101, 92},  Vec3b{143, 100, 91},
       Vec3b{139, 99, 98},   Vec3b{141, 100, 99},  Vec3b{142, 101, 100},
       Vec3b{141, 100, 99},  Vec3b{143, 102, 101}, Vec3b{143, 102, 101},
       Vec3b{143, 102, 101}, Vec3b{144, 103, 102}, Vec3b{140, 102, 109},
       Vec3b{141, 103, 110}, Vec3b{141, 103, 110}, Vec3b{143, 104, 112},
       Vec3b{143, 104, 112}, Vec3b{143, 104, 112}, Vec3b{144, 105, 113},
       Vec3b{143, 104, 112}, Vec3b{138, 103, 121}, Vec3b{140, 105, 123},
       Vec3b{141, 105, 123}, Vec3b{140, 105, 122}, Vec3b{141, 105, 123},
       Vec3b{141, 106, 124}, Vec3b{142, 106, 124}, Vec3b{142, 107, 124},
       Vec3b{142, 107, 124}, Vec3b{138, 106, 133}, Vec3b{138, 105, 132},
       Vec3b{139, 107, 134}, Vec3b{139, 106, 133}, Vec3b{140, 107, 134},
       Vec3b{141, 108, 135}, Vec3b{138, 108, 144}, Vec3b{137, 106, 143},
       Vec3b{137, 107, 144}, Vec3b{137, 107, 144}, Vec3b{138, 107, 144},
       Vec3b{138, 108, 145}, Vec3b{134, 107, 151}, Vec3b{134, 107, 152},
       Vec3b{136, 109, 154}, Vec3b{136, 109, 154}, Vec3b{137, 110, 155},
       Vec3b{136, 109, 153}, Vec3b{135, 109, 163}, Vec3b{135, 109, 164},
       Vec3b{137, 111, 166}, Vec3b{134, 109, 163}, Vec3b{135, 109, 164},
       Vec3b{130, 109, 173}, Vec3b{131, 109, 173}, Vec3b{131, 110, 174},
       Vec3b{132, 110, 174}, Vec3b{132, 111, 175}, Vec3b{133, 111, 176},
       Vec3b{133, 112, 176}, Vec3b{134, 112, 177}, Vec3b{127, 110, 188},
       Vec3b{128, 111, 189}, Vec3b{130, 113, 191}, Vec3b{129, 112, 190},
       Vec3b{129, 112, 191}, Vec3b{130, 113, 192}, Vec3b{131, 114, 193},
       Vec3b{131, 114, 193}, Vec3b{132, 115, 194}, Vec3b{133, 115, 195},
       Vec3b{125, 113, 210}, Vec3b{126, 113, 211}, Vec3b{127, 114, 213},
       Vec3b{128, 115, 214}, Vec3b{128, 116, 215}, Vec3b{129, 117, 216},
       Vec3b{130, 117, 217}, Vec3b{131, 118, 218}, Vec3b{131, 118, 218},
       Vec3b{133, 120, 221}, Vec3b{133, 120, 221}, Vec3b{124, 117, 238},
       Vec3b{125, 118, 240}, Vec3b{125, 118, 240}, Vec3b{126, 119, 242},
       Vec3b{128, 122, 246}, Vec3b{130, 123, 247}, Vec3b{130, 123, 248},
       Vec3b{129, 122, 247}, Vec3b{130, 124, 249}, Vec3b{134, 127, 254},
       Vec3b{133, 127, 254}, Vec3b{135, 128, 255}, Vec3b{137, 130, 255},
       Vec3b{138, 132, 255}, Vec3b{125, 129, 255}, Vec3b{127, 131, 255},
       Vec3b{125, 129, 255}, Vec3b{127, 131, 255}, Vec3b{129, 133, 255},
       Vec3b{131, 135, 255}, Vec3b{133, 138, 255}, Vec3b{133, 138, 255},
       Vec3b{134, 138, 255}, Vec3b{136, 141, 255}, Vec3b{137, 141, 255},
       Vec3b{138, 142, 255}, Vec3b{140, 145, 255}, Vec3b{143, 147, 255},
       Vec3b{145, 150, 255}, Vec3b{145, 150, 255}, Vec3b{148, 152, 255},
       Vec3b{148, 153, 255}, Vec3b{151, 156, 255}, Vec3b{138, 153, 255},
       Vec3b{138, 153, 255}, Vec3b{141, 156, 255}, Vec3b{144, 160, 255},
       Vec3b{146, 162, 255}, Vec3b{147, 163, 255}, Vec3b{149, 165, 255},
       Vec3b{153, 169, 255}, Vec3b{154, 171, 255}, Vec3b{154, 170, 255},
       Vec3b{158, 174, 255}, Vec3b{162, 179, 255}, Vec3b{161, 178, 255},
       Vec3b{165, 182, 255}, Vec3b{169, 187, 255}, Vec3b{169, 186, 255},
       Vec3b{171, 189, 255}, Vec3b{176, 194, 255}, Vec3b{176, 194, 255},
       Vec3b{181, 199, 255}, Vec3b{181, 199, 255}, Vec3b{185, 203, 255},
       Vec3b{186, 205, 255}});
  return image;
}
}  // namespace quantized_image