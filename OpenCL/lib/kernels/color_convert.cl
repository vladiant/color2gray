// color_convert.cl — RGB to LAB color space conversion kernel
// Each work-item converts one pixel.

__kernel void rgb_to_lab(__global const unsigned char* rgb_in,
                         __global float* lab_out,
                         const int N) {
  const int i = get_global_id(0);
  if (i >= N) return;

  // Read RGB values
  const float R = rgb_in[3 * i]     / 255.0f;
  const float G = rgb_in[3 * i + 1] / 255.0f;
  const float B = rgb_in[3 * i + 2] / 255.0f;

  // RGB -> XYZ (sRGB matrix)
  float X = 0.412453f * R + 0.357580f * G + 0.180423f * B;
  float Y = 0.212671f * R + 0.715160f * G + 0.072169f * B;
  float Z = 0.019334f * R + 0.119193f * G + 0.950227f * B;

  // Normalize to D65 illuminant
  const float Xn = 0.9513f;
  const float Yn = 1.000f;
  const float Zn = 1.0886f;

  X = X / Xn;
  Y = Y / Yn;
  Z = Z / Zn;

  // XYZ -> LAB
  const float one_third = 0.3333333333f;
  const float X_third = pow(X, one_third);
  const float Y_third = pow(Y, one_third);
  const float Z_third = pow(Z, one_third);

  float l;
  if (Y > 0.008856f)
    l = 116.0f * Y_third - 16.0f;
  else
    l = 903.3f * Y;

  const float a = 500.0f * (X_third - Y_third);
  const float b = 200.0f * (Y_third - Z_third);

  // Output: packed as (l, a, b) per pixel
  lab_out[3 * i]     = l;
  lab_out[3 * i + 1] = a;
  lab_out[3 * i + 2] = b;
}
