// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Speed-critical encoding functions.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdlib.h>  // for abs()

#include "src/dsp/dsp.h"
#include "src/enc/vp8i_enc.h"

static WEBP_INLINE uint8_t clip_8b(int v) {
  return (!(v & ~0xff)) ? v : (v < 0) ? 0 : 255;
}

#if !WEBP_NEON_OMIT_C_CODE
static WEBP_INLINE int clip_max(int v, int max) {
  return (v > max) ? max : v;
}
#endif  // !WEBP_NEON_OMIT_C_CODE

//------------------------------------------------------------------------------
// Compute susceptibility based on DCT-coeff histograms:
// the higher, the "easier" the macroblock is to compress.

const int VP8DspScan[16 + 4 + 4] = {
  // Luma
  0 +  0 * BPS,  4 +  0 * BPS, 8 +  0 * BPS, 12 +  0 * BPS,
  0 +  4 * BPS,  4 +  4 * BPS, 8 +  4 * BPS, 12 +  4 * BPS,
  0 +  8 * BPS,  4 +  8 * BPS, 8 +  8 * BPS, 12 +  8 * BPS,
  0 + 12 * BPS,  4 + 12 * BPS, 8 + 12 * BPS, 12 + 12 * BPS,

  0 + 0 * BPS,   4 + 0 * BPS, 0 + 4 * BPS,  4 + 4 * BPS,    // U
  8 + 0 * BPS,  12 + 0 * BPS, 8 + 4 * BPS, 12 + 4 * BPS     // V
};

// general-purpose util function
void VP8SetHistogramData(const int distribution[MAX_COEFF_THRESH + 1],
                         VP8Histogram* const histo) {
  int max_value = 0, last_non_zero = 1;
  int k;
  for (k = 0; k <= MAX_COEFF_THRESH; ++k) {
    const int value = distribution[k];
    if (value > 0) {
      if (value > max_value) max_value = value;
      last_non_zero = k;
    }
  }
  histo->max_value = max_value;
  histo->last_non_zero = last_non_zero;
}

#if !WEBP_NEON_OMIT_C_CODE
static void CollectHistogram_C(const uint8_t* ref, const uint8_t* pred,
                               int start_block, int end_block,
                               VP8Histogram* const histo) {
  int j;
  int distribution[MAX_COEFF_THRESH + 1] = { 0 };
  for (j = start_block; j < end_block; ++j) {
    int k;
    int16_t out[16];

    VP8FTransform(ref + VP8DspScan[j], pred + VP8DspScan[j], out);

    // Convert coefficients to bin.
    for (k = 0; k < 16; ++k) {
      const int v = abs(out[k]) >> 3;
      const int clipped_value = clip_max(v, MAX_COEFF_THRESH);
      ++distribution[clipped_value];
    }
  }
  VP8SetHistogramData(distribution, histo);
}
#endif  // !WEBP_NEON_OMIT_C_CODE

//------------------------------------------------------------------------------
// run-time tables (~4k)

static uint8_t clip1[255 + 510 + 1];    // clips [-255,510] to [0,255]

// We declare this variable 'volatile' to prevent instruction reordering
// and make sure it's set to true _last_ (so as to be thread-safe)
static volatile int tables_ok = 0;

static WEBP_TSAN_IGNORE_FUNCTION void InitTables(void) {
  if (!tables_ok) {
    int i;
    for (i = -255; i <= 255 + 255; ++i) {
      clip1[255 + i] = clip_8b(i);
    }
    tables_ok = 1;
  }
}


//------------------------------------------------------------------------------
// Transforms (Paragraph 14.4)

#if !WEBP_NEON_OMIT_C_CODE

#define STORE(x, y, v) \
  dst[(x) + (y) * BPS] = clip_8b(ref[(x) + (y) * BPS] + ((v) >> 3))

static const int kC1 = 20091 + (1 << 16);
static const int kC2 = 35468;
#define MUL(a, b) (((a) * (b)) >> 16)

static WEBP_INLINE void ITransformOne(const uint8_t* ref, const int16_t* in,
                                      uint8_t* dst) {
  int C[4 * 4], *tmp;
  int i;
  tmp = C;
  for (i = 0; i < 4; ++i) {    // vertical pass
#pragma HLS unroll
    const int a = in[0] + in[8];
    const int b = in[0] - in[8];
    const int c = MUL(in[4], kC2) - MUL(in[12], kC1);
    const int d = MUL(in[4], kC1) + MUL(in[12], kC2);
    tmp[0] = a + d;
    tmp[1] = b + c;
    tmp[2] = b - c;
    tmp[3] = a - d;
    tmp += 4;
    in++;
  }

  tmp = C;
  for (i = 0; i < 4; ++i) {    // horizontal pass
#pragma HLS unroll
    const int dc = tmp[0] + 4;
    const int a =  dc +  tmp[8];
    const int b =  dc -  tmp[8];
    const int c = MUL(tmp[4], kC2) - MUL(tmp[12], kC1);
    const int d = MUL(tmp[4], kC1) + MUL(tmp[12], kC2);
    STORE(0, i, a + d);
    STORE(1, i, b + c);
    STORE(2, i, b - c);
    STORE(3, i, a - d);
    tmp++;
  }
}
static void ITransform_C(const uint8_t* ref, const int16_t* in, uint8_t* dst,
                         int do_two) {
  ITransformOne(ref, in, dst);
  if (do_two) {
    ITransformOne(ref + 4, in + 16, dst + 4);
  }
}

static void FTransform_C(const uint8_t* src, const uint8_t* ref, int16_t* out) {
  int i;
  int tmp[16];
  for (i = 0; i < 4; ++i, src += 4, ref += 4) {
    const int d0 = src[0] - ref[0];   // 9bit dynamic range ([-255,255])
    const int d1 = src[1] - ref[1];
    const int d2 = src[2] - ref[2];
    const int d3 = src[3] - ref[3];
    const int a0 = (d0 + d3);         // 10b                      [-510,510]
    const int a1 = (d1 + d2);
    const int a2 = (d1 - d2);
    const int a3 = (d0 - d3);
    tmp[0 + i * 4] = (a0 + a1) * 8;   // 14b                      [-8160,8160]
    tmp[1 + i * 4] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;      // [-7536,7542]
    tmp[2 + i * 4] = (a0 - a1) * 8;
    tmp[3 + i * 4] = (a3 * 2217 - a2 * 5352 +  937) >> 9;
  }
  for (i = 0; i < 4; ++i) {
    const int a0 = (tmp[0 + i] + tmp[12 + i]);  // 15b
    const int a1 = (tmp[4 + i] + tmp[ 8 + i]);
    const int a2 = (tmp[4 + i] - tmp[ 8 + i]);
    const int a3 = (tmp[0 + i] - tmp[12 + i]);
    out[0 + i] = (a0 + a1 + 7) >> 4;            // 12b
    out[4 + i] = ((a2 * 2217 + a3 * 5352 + 12000) >> 16) + (a3 != 0);
    out[8 + i] = (a0 - a1 + 7) >> 4;
    out[12+ i] = ((a3 * 2217 - a2 * 5352 + 51000) >> 16);
  }
}
#endif  // !WEBP_NEON_OMIT_C_CODE

static void FTransform2_C(const uint8_t* src, const uint8_t* ref,
                          int16_t* out) {
  VP8FTransform(src, ref, out);
  VP8FTransform(src + 4, ref + 4, out + 16);
}

#if !WEBP_NEON_OMIT_C_CODE
static void FTransformWHT_C(const int16_t* in, int16_t* out) {
  // input is 12b signed
  int32_t tmp[16];
  int i;
  for (i = 0; i < 4; ++i, in += 4) {
    const int a0 = (in[0] + in[2]);  // 13b
    const int a1 = (in[1] + in[3]);
    const int a2 = (in[1] - in[3]);
    const int a3 = (in[0] - in[2]);
    tmp[0 + i * 4] = a0 + a1;   // 14b
    tmp[1 + i * 4] = a3 + a2;
    tmp[2 + i * 4] = a3 - a2;
    tmp[3 + i * 4] = a0 - a1;
  }
  for (i = 0; i < 4; ++i) {
    const int a0 = (tmp[0 + i] + tmp[8 + i]);  // 15b
    const int a1 = (tmp[4 + i] + tmp[12+ i]);
    const int a2 = (tmp[4 + i] - tmp[12+ i]);
    const int a3 = (tmp[0 + i] - tmp[8 + i]);
    const int b0 = a0 + a1;    // 16b
    const int b1 = a3 + a2;
    const int b2 = a3 - a2;
    const int b3 = a0 - a1;
    out[ 0 + i] = b0 >> 1;     // 15b
    out[ 4 + i] = b1 >> 1;
    out[ 8 + i] = b2 >> 1;
    out[12 + i] = b3 >> 1;
  }
}
#endif  // !WEBP_NEON_OMIT_C_CODE

#undef MUL
#undef STORE

//------------------------------------------------------------------------------
// Intra predictions

static WEBP_INLINE void Fill(uint8_t* dst, int value, int size) {
  int i,j;
  for (j = 0; j < size; ++j) {
    for(i = 0; i < size; ++i){
        dst[j * size + i] = value;
    }
  }
}

static WEBP_INLINE void VerticalPred(uint8_t* dst, uint8_t* top, int size) {
  int i,j;
    for (j = 0; j < size; ++j) {
    	for(i = 0; i < size; ++i){
    		dst[j * size + i] = top[i];
    	}
    }
}

static WEBP_INLINE void HorizontalPred(uint8_t* dst, uint8_t* left, int size) {
    int i,j;
    for (j = 0; j < size; ++j) {
    	for(i = 0; i < size; ++i){
    		dst[j * size + i] = left[j];
    	}
    }
}

static WEBP_INLINE void TrueMotion(uint8_t* dst, uint8_t* left, uint8_t* top, uint8_t top_left, int size, int x, int y) {
  int i,j;
  int tmp;
  if (x != 0) {
    if (y != 0) {
      for (j = 0; j < size; ++j) {
        for (i = 0; i < size; ++i) {
        	tmp = top[i] + left[j] - top_left;
        	dst[j * size + i] = (!(tmp & ~0xff)) ? (uint8_t)tmp : (tmp < 0) ? 0 : 255;
        }
      }
    } else {
      HorizontalPred(dst, left, size);
    }
  } else {
    // true motion without left samples (hence: with default 129 value)
    // is equivalent to VE prediction where you just copy the top samples.
    // Note that if top samples are not available, the default value is
    // then 129, and not 127 as in the VerticalPred case.
    if (y != 0) {
      VerticalPred(dst, top, size);
    } else {
      Fill(dst, 129, size);
    }
  }
}

static WEBP_INLINE void DCMode(uint8_t* dst, uint8_t* left, uint8_t* top,
                               int size, int round, int shift) {
  int DC = 0;
  int j;
	for (j = 0; j < size; ++j){
		DC += top[j] + left[j];
	}
  DC = (DC + round) >> shift;
  Fill(dst, DC, size);
}

//------------------------------------------------------------------------------
// Chroma 8x8 prediction (paragraph 12.2)

void IntraChromaPreds_C(
		uint8_t UVPred[8][8*8],
        uint8_t left_u[8], uint8_t top_u[8], uint8_t top_left_u,
		uint8_t left_v[8], uint8_t top_v[8], uint8_t top_left_v,
		int x, int y) {
  // U block
  DCMode(UVPred[0], left_u, top_u, 8, 8, 4);
  VerticalPred(UVPred[1], top_u, 8);
  HorizontalPred(UVPred[2], left_u, 8);
  TrueMotion(UVPred[3], left_u, top_u, top_left_u, 8, x, y);
  // V block
  DCMode(UVPred[4], left_v, top_v, 8, 8, 4);
  VerticalPred(UVPred[5], top_v, 8);
  HorizontalPred(UVPred[6], left_v, 8);
  TrueMotion(UVPred[7], left_v, top_v, top_left_v, 8, x, y);
}


//------------------------------------------------------------------------------
// luma 16x16 prediction (paragraph 12.3)

void Intra16Preds_C(uint8_t YPred[4][16*16], uint8_t left_y[16],
		uint8_t* top_y, uint8_t top_left_y, int x, int y) {
  DCMode(YPred[0], left_y, top_y, 16, 16, 5);
  VerticalPred(YPred[1], top_y, 16);
  HorizontalPred(YPred[2], left_y, 16);
  TrueMotion(YPred[3], left_y, top_y, top_left_y, 16, x, y);
}


//------------------------------------------------------------------------------
// luma 4x4 prediction

#define DST(x, y) dst[(x) + (y) * BPS]
#define AVG3(a, b, c) ((uint8_t)(((a) + 2 * (b) + (c) + 2) >> 2))
#define AVG2(a, b) (((a) + (b) + 1) >> 1)

static void VE4(uint8_t* dst, uint8_t top_left, uint8_t* top, uint8_t* top_right) {    // vertical
  uint8_t vals[4] = {
    AVG3(top_left, top[0], top[1]),
    AVG3(top[0], top[1], top[2]),
    AVG3(top[1], top[2], top[3]),
    AVG3(top[2], top[3], top_right[0])
  };
  int i,j;
    for (j = 0; j < 4; ++j) {
    	for(i = 0; i < 4; ++i){
    		dst[j * 4 + i] = vals[i];
    	}
    }
}

static void HE4(uint8_t* dst, uint8_t* left, uint8_t top_left) {    // horizontal
  const int X = top_left;
  const int I = left[0];
  const int J = left[1];
  const int K = left[2];
  const int L = left[3];
  ((uint32_t*)dst)[0] = 0x01010101U * AVG3(X, I, J);
  ((uint32_t*)dst)[1] = 0x01010101U * AVG3(I, J, K);
  ((uint32_t*)dst)[2] = 0x01010101U * AVG3(J, K, L);
  ((uint32_t*)dst)[3] = 0x01010101U * AVG3(K, L, L);
}


static void DC4(uint8_t* dst, uint8_t* top, uint8_t* left) {
  uint32_t dc = 4;
  int i;
  for (i = 0; i < 4; ++i){
	  dc += top[i] + left[i];
  }
  dc = dc >> 3;
  Fill(dst, dc, 4);
}

static void RD4(uint8_t* dst, uint8_t* left, uint8_t top_left, uint8_t* top) {
  const int X = top_left;
  const int I = left[0];
  const int J = left[1];
  const int K = left[2];
  const int L = left[3];
  const int A = top[0];
  const int B = top[1];
  const int C = top[2];
  const int D = top[3];
  DST(0, 3)                                     = AVG3(J, K, L);
  DST(0, 2) = DST(1, 3)                         = AVG3(I, J, K);
  DST(0, 1) = DST(1, 2) = DST(2, 3)             = AVG3(X, I, J);
  DST(0, 0) = DST(1, 1) = DST(2, 2) = DST(3, 3) = AVG3(A, X, I);
  DST(1, 0) = DST(2, 1) = DST(3, 2)             = AVG3(B, A, X);
  DST(2, 0) = DST(3, 1)                         = AVG3(C, B, A);
  DST(3, 0)                                     = AVG3(D, C, B);
}

static void LD4(uint8_t* dst, uint8_t* top, uint8_t* top_right) {
  const int A = top[0];
  const int B = top[1];
  const int C = top[2];
  const int D = top[3];
  const int E = top_right[0];
  const int F = top_right[1];
  const int G = top_right[2];
  const int H = top_right[3];
  DST(0, 0)                                     = AVG3(A, B, C);
  DST(1, 0) = DST(0, 1)                         = AVG3(B, C, D);
  DST(2, 0) = DST(1, 1) = DST(0, 2)             = AVG3(C, D, E);
  DST(3, 0) = DST(2, 1) = DST(1, 2) = DST(0, 3) = AVG3(D, E, F);
  DST(3, 1) = DST(2, 2) = DST(1, 3)             = AVG3(E, F, G);
  DST(3, 2) = DST(2, 3)                         = AVG3(F, G, H);
  DST(3, 3)                                     = AVG3(G, H, H);
}

static void VR4(uint8_t* dst, uint8_t* left, uint8_t top_left, uint8_t* top) {
  const int X = top_left;
  const int I = left[0];
  const int J = left[1];
  const int K = left[2];
  const int A = top[0];
  const int B = top[1];
  const int C = top[2];
  const int D = top[3];
  DST(0, 0) = DST(1, 2) = AVG2(X, A);
  DST(1, 0) = DST(2, 2) = AVG2(A, B);
  DST(2, 0) = DST(3, 2) = AVG2(B, C);
  DST(3, 0)             = AVG2(C, D);

  DST(0, 3) =             AVG3(K, J, I);
  DST(0, 2) =             AVG3(J, I, X);
  DST(0, 1) = DST(1, 3) = AVG3(I, X, A);
  DST(1, 1) = DST(2, 3) = AVG3(X, A, B);
  DST(2, 1) = DST(3, 3) = AVG3(A, B, C);
  DST(3, 1) =             AVG3(B, C, D);
}

static void VL4(uint8_t* dst, uint8_t* top, uint8_t* top_right) {
  const int A = top[0];
  const int B = top[1];
  const int C = top[2];
  const int D = top[3];
  const int E = top_right[0];
  const int F = top_right[1];
  const int G = top_right[2];
  const int H = top_right[3];
  DST(0, 0) =             AVG2(A, B);
  DST(1, 0) = DST(0, 2) = AVG2(B, C);
  DST(2, 0) = DST(1, 2) = AVG2(C, D);
  DST(3, 0) = DST(2, 2) = AVG2(D, E);

  DST(0, 1) =             AVG3(A, B, C);
  DST(1, 1) = DST(0, 3) = AVG3(B, C, D);
  DST(2, 1) = DST(1, 3) = AVG3(C, D, E);
  DST(3, 1) = DST(2, 3) = AVG3(D, E, F);
              DST(3, 2) = AVG3(E, F, G);
              DST(3, 3) = AVG3(F, G, H);
}

static void HU4(uint8_t* dst, uint8_t* left) {
  const int I = left[0];
  const int J = left[1];
  const int K = left[2];
  const int L = left[3];
  DST(0, 0) =             AVG2(I, J);
  DST(2, 0) = DST(0, 1) = AVG2(J, K);
  DST(2, 1) = DST(0, 2) = AVG2(K, L);
  DST(1, 0) =             AVG3(I, J, K);
  DST(3, 0) = DST(1, 1) = AVG3(J, K, L);
  DST(3, 1) = DST(1, 2) = AVG3(K, L, L);
  DST(3, 2) = DST(2, 2) =
  DST(0, 3) = DST(1, 3) = DST(2, 3) = DST(3, 3) = L;
}

static void HD4(uint8_t* dst, uint8_t* left, uint8_t top_left, uint8_t* top) {
  const int X = top_left;
  const int I = left[0];
  const int J = left[1];
  const int K = left[2];
  const int L = left[3];
  const int A = top[0];
  const int B = top[1];
  const int C = top[2];

  DST(0, 0) = DST(2, 1) = AVG2(I, X);
  DST(0, 1) = DST(2, 2) = AVG2(J, I);
  DST(0, 2) = DST(2, 3) = AVG2(K, J);
  DST(0, 3)             = AVG2(L, K);

  DST(3, 0)             = AVG3(A, B, C);
  DST(2, 0)             = AVG3(X, A, B);
  DST(1, 0) = DST(3, 1) = AVG3(I, X, A);
  DST(1, 1) = DST(3, 2) = AVG3(J, I, X);
  DST(1, 2) = DST(3, 3) = AVG3(K, J, I);
  DST(1, 3)             = AVG3(L, K, J);
}

static void TM4(uint8_t* dst, uint8_t* top, uint8_t* left, uint8_t top_left) {
  int i, j;
  int tmp;
  for (j = 0; j < 4; ++j) {
    for (i = 0; i < 4; ++i) {
      tmp = top[i] + left[j] - top_left;
      dst[j * 4 + i] = (tmp>0xff) ? 0xff : (tmp<0) ? 0 : (uint8_t)tmp;
    }
  }
}

#undef DST
#undef AVG3
#undef AVG2

// Left samples are top[-5 .. -2], top_left is top[-1], top are
// located at top[0..3], and top right is top[4..7]
static void Intra4Preds_C(
		uint8_t Pred[10][16], uint8_t left[4], uint8_t top_left, uint8_t top[4], uint8_t top_right[4]) {
  DC4(Pred[0], top, left);
  TM4(Pred[1], top, left, top_left);
  VE4(Pred[2], top_left, top, top_right);
  HE4(Pred[3], left, top_left);
  RD4(Pred[4], left, top_left, top);
  VR4(Pred[5], left, top_left, top);
  LD4(Pred[6], top, top_right);
  VL4(Pred[7], top, top_right);
  HD4(Pred[8], left, top_left, top);
  HU4(Pred[9], top);
}

//------------------------------------------------------------------------------
// Metric

#if !WEBP_NEON_OMIT_C_CODE
static WEBP_INLINE int GetSSE(const uint8_t* a, const uint8_t* b,
                              int w, int h) {
  int count = 0;
  int y, x;
  for (y = 0; y < h; ++y) {
    for (x = 0; x < w; ++x) {
      const int diff = (int)a[x + y * w] - b[x + y * w];
      count += diff * diff;
    }
  }
  return count;
}

static int SSE16x16_C(const uint8_t* a, const uint8_t* b) {
  return GetSSE(a, b, 16, 16);
}
static int SSE16x8_C(const uint8_t* a, const uint8_t* b) {
  return GetSSE(a, b, 16, 8);
}
static int SSE8x8_C(const uint8_t* a, const uint8_t* b) {
  return GetSSE(a, b, 8, 8);
}
static int SSE4x4_C(const uint8_t* a, const uint8_t* b) {
  return GetSSE(a, b, 4, 4);
}
#endif  // !WEBP_NEON_OMIT_C_CODE

static void Mean16x4_C(const uint8_t* ref, uint32_t dc[4]) {
  int k, x, y;
  for (k = 0; k < 4; ++k) {
    uint32_t avg = 0;
    for (y = 0; y < 4; ++y) {
      for (x = 0; x < 4; ++x) {
        avg += ref[x + y * BPS];
      }
    }
    dc[k] = avg;
    ref += 4;   // go to next 4x4 block.
  }
}

//------------------------------------------------------------------------------
// Texture distortion
//
// We try to match the spectral content (weighted) between source and
// reconstructed samples.

#if !WEBP_NEON_OMIT_C_CODE
// Hadamard transform
// Returns the weighted sum of the absolute value of transformed coefficients.
// w[] contains a row-major 4 by 4 symmetric matrix.
static int TTransform(const uint8_t* in, const uint16_t* w) {
  int sum = 0;
  int tmp[16];
  int i;
  // horizontal pass
  for (i = 0; i < 4; ++i, in += 4) {
    const int a0 = in[0] + in[2];
    const int a1 = in[1] + in[3];
    const int a2 = in[1] - in[3];
    const int a3 = in[0] - in[2];
    tmp[0 + i * 4] = a0 + a1;
    tmp[1 + i * 4] = a3 + a2;
    tmp[2 + i * 4] = a3 - a2;
    tmp[3 + i * 4] = a0 - a1;
  }
  // vertical pass
  for (i = 0; i < 4; ++i, ++w) {
    const int a0 = tmp[0 + i] + tmp[8 + i];
    const int a1 = tmp[4 + i] + tmp[12+ i];
    const int a2 = tmp[4 + i] - tmp[12+ i];
    const int a3 = tmp[0 + i] - tmp[8 + i];
    const int b0 = a0 + a1;
    const int b1 = a3 + a2;
    const int b2 = a3 - a2;
    const int b3 = a0 - a1;

    sum += w[ 0] * abs(b0);
    sum += w[ 4] * abs(b1);
    sum += w[ 8] * abs(b2);
    sum += w[12] * abs(b3);
  }
  return sum;
}


static int Disto4x4_C(const uint8_t* const a, const uint8_t* const b,
                      const uint16_t* const w) {
  const int sum1 = TTransform(a, w);
  const int sum2 = TTransform(b, w);
  return abs(sum2 - sum1) >> 5;
}


static int Disto16x16_C(const uint8_t* const a, const uint8_t* const b,
                        const uint16_t* const w) {
  int D = 0;

  uint8_t tmp_a[16][16], tmp_b[16][16];

  const uint16_t VP8Scan[16] = {
    0 +  0 * 16,  4 +  0 * 16, 8 +  0 * 16, 12 +  0 * 16,
    0 +  4 * 16,  4 +  4 * 16, 8 +  4 * 16, 12 +  4 * 16,
    0 +  8 * 16,  4 +  8 * 16, 8 +  8 * 16, 12 +  8 * 16,
    0 + 12 * 16,  4 + 12 * 16, 8 + 12 * 16, 12 + 12 * 16,
  };

  int i,j,n;
  for(n = 0; n < 16; n++){
	  for(j = 0; j < 4; j++){
		  for(i = 0; i < 4; i++){
			  tmp_a[n][j * 4 + i] = a[VP8Scan[n] + j * 16 + i];
			  tmp_b[n][j * 4 + i] = b[VP8Scan[n] + j * 16 + i];
		  }
	  }
  }


    for (i = 0; i < 16; i++) {
      D += Disto4x4_C(tmp_a[i], tmp_b[i], w);
    }

  return D;

}

#endif  // !WEBP_NEON_OMIT_C_CODE

//------------------------------------------------------------------------------
// Quantization
//

static const uint8_t kZigzag[16] = {
  0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15
};

// Simple quantization
static int QuantizeBlock_C(int16_t* in, int16_t* out,
                           const VP8Matrix* const mtx) {
  int last = -1;
  int n;
  int16_t in_tmp[16];
  for (n = 0; n < 16; ++n) {
    const int j = kZigzag[n];
    const int sign = (in[j] < 0);
    const uint32_t coeff = (sign ? -in[j] : in[j]) + mtx->sharpen_[j];
    if (coeff > mtx->zthresh_[j]) {
      const uint32_t Q = mtx->q_[j];
      const uint32_t iQ = mtx->iq_[j];
      const uint32_t B = mtx->bias_[j];
      int level = QUANTDIV(coeff, iQ, B);
      if (level > MAX_LEVEL) level = MAX_LEVEL;
      if (sign) level = -level;
      in_tmp[j] = level * (int)Q;
      out[n] = level;
      if (level) last = n;
    } else {
      out[n] = 0;
      in_tmp[j] = 0;
    }
  }
  for (n = 0; n < 16; ++n) {
	in[n] = in_tmp[n];
  }
  return (last >= 0);
}

#if !WEBP_NEON_OMIT_C_CODE || WEBP_NEON_WORK_AROUND_GCC
static int Quantize2Blocks_C(int16_t in[32], int16_t out[32],
                             const VP8Matrix* const mtx) {
  int nz;
  nz  = VP8EncQuantizeBlock(in + 0 * 16, out + 0 * 16, mtx) << 0;
  nz |= VP8EncQuantizeBlock(in + 1 * 16, out + 1 * 16, mtx) << 1;
  return nz;
}
#endif  // !WEBP_NEON_OMIT_C_CODE || WEBP_NEON_WORK_AROUND_GCC

//------------------------------------------------------------------------------
// Block copy

static WEBP_INLINE void Copy(const uint8_t* src, uint8_t* dst, int w, int h) {
  int y;
  for (y = 0; y < h; ++y) {
    memcpy(dst, src, w);
    src += BPS;
    dst += BPS;
  }
}

static void Copy4x4_C(const uint8_t* src, uint8_t* dst) {
  Copy(src, dst, 4, 4);
}

static void Copy16x8_C(const uint8_t* src, uint8_t* dst) {
  Copy(src, dst, 16, 8);
}

//------------------------------------------------------------------------------
// Initialization

// Speed-critical function pointers. We have to initialize them to the default
// implementations within VP8EncDspInit().
VP8CHisto VP8CollectHistogram;
VP8Idct VP8ITransform;
VP8Fdct VP8FTransform;
VP8Fdct VP8FTransform2;
VP8WHT VP8FTransformWHT;
VP8Intra4Preds VP8EncPredLuma4;
VP8IntraPreds VP8EncPredLuma16;
VP8IntraPreds VP8EncPredChroma8;
VP8Metric VP8SSE16x16;
VP8Metric VP8SSE8x8;
VP8Metric VP8SSE16x8;
VP8Metric VP8SSE4x4;
VP8WMetric VP8TDisto4x4;
VP8WMetric VP8TDisto16x16;
VP8MeanMetric VP8Mean16x4;
VP8QuantizeBlock VP8EncQuantizeBlock;
VP8Quantize2Blocks VP8EncQuantize2Blocks;
VP8QuantizeBlockWHT VP8EncQuantizeBlockWHT;
VP8BlockCopy VP8Copy4x4;
VP8BlockCopy VP8Copy16x8;

extern void VP8EncDspInitSSE2(void);
extern void VP8EncDspInitSSE41(void);
extern void VP8EncDspInitAVX2(void);
extern void VP8EncDspInitNEON(void);
extern void VP8EncDspInitMIPS32(void);
extern void VP8EncDspInitMIPSdspR2(void);
extern void VP8EncDspInitMSA(void);

WEBP_DSP_INIT_FUNC(VP8EncDspInit) {
  VP8DspInit();  // common inverse transforms
  InitTables();

  // default C implementations
#if !WEBP_NEON_OMIT_C_CODE
  VP8ITransform = ITransform_C;
  VP8FTransform = FTransform_C;
  VP8FTransformWHT = FTransformWHT_C;
  VP8TDisto4x4 = Disto4x4_C;
  VP8TDisto16x16 = Disto16x16_C;
  VP8CollectHistogram = CollectHistogram_C;
  VP8SSE16x16 = SSE16x16_C;
  VP8SSE16x8 = SSE16x8_C;
  VP8SSE8x8 = SSE8x8_C;
  VP8SSE4x4 = SSE4x4_C;
#endif

#if !WEBP_NEON_OMIT_C_CODE || WEBP_NEON_WORK_AROUND_GCC
  VP8EncQuantizeBlock = QuantizeBlock_C;
  VP8EncQuantize2Blocks = Quantize2Blocks_C;
#endif

  VP8FTransform2 = FTransform2_C;
  VP8EncPredLuma4 = Intra4Preds_C;
  VP8EncPredLuma16 = Intra16Preds_C;
  VP8EncPredChroma8 = IntraChromaPreds_C;
  VP8Mean16x4 = Mean16x4_C;
  VP8EncQuantizeBlockWHT = QuantizeBlock_C;
  VP8Copy4x4 = Copy4x4_C;
  VP8Copy16x8 = Copy16x8_C;

  // If defined, use CPUInfo() to overwrite some pointers with faster versions.
  if (VP8GetCPUInfo != NULL) {
#if defined(WEBP_USE_SSE2)
    if (VP8GetCPUInfo(kSSE2)) {
      VP8EncDspInitSSE2();
#if defined(WEBP_USE_SSE41)
      if (VP8GetCPUInfo(kSSE4_1)) {
        VP8EncDspInitSSE41();
      }
#endif
    }
#endif
#if defined(WEBP_USE_AVX2)
    if (VP8GetCPUInfo(kAVX2)) {
      VP8EncDspInitAVX2();
    }
#endif
#if defined(WEBP_USE_MIPS32)
    if (VP8GetCPUInfo(kMIPS32)) {
      VP8EncDspInitMIPS32();
    }
#endif
#if defined(WEBP_USE_MIPS_DSP_R2)
    if (VP8GetCPUInfo(kMIPSdspR2)) {
      VP8EncDspInitMIPSdspR2();
    }
#endif
#if defined(WEBP_USE_MSA)
    if (VP8GetCPUInfo(kMSA)) {
      VP8EncDspInitMSA();
    }
#endif
  }

#if defined(WEBP_USE_NEON)
  if (WEBP_NEON_OMIT_C_CODE ||
      (VP8GetCPUInfo != NULL && VP8GetCPUInfo(kNEON))) {
    VP8EncDspInitNEON();
  }
#endif

  assert(VP8ITransform != NULL);
  assert(VP8FTransform != NULL);
  assert(VP8FTransformWHT != NULL);
  assert(VP8TDisto4x4 != NULL);
  assert(VP8TDisto16x16 != NULL);
  assert(VP8CollectHistogram != NULL);
  assert(VP8SSE16x16 != NULL);
  assert(VP8SSE16x8 != NULL);
  assert(VP8SSE8x8 != NULL);
  assert(VP8SSE4x4 != NULL);
  assert(VP8EncQuantizeBlock != NULL);
  assert(VP8EncQuantize2Blocks != NULL);
  assert(VP8FTransform2 != NULL);
  assert(VP8EncPredLuma4 != NULL);
  assert(VP8EncPredLuma16 != NULL);
  assert(VP8EncPredChroma8 != NULL);
  assert(VP8Mean16x4 != NULL);
  assert(VP8EncQuantizeBlockWHT != NULL);
  assert(VP8Copy4x4 != NULL);
  assert(VP8Copy16x8 != NULL);
}
