// Gated DeltaNet 15M single-kernel decoder.
//
// Two builds share this file:
//   * BUILD_DECODE_KERNEL  -> the HLS `decode` top for Vitis HLS. A single
//     autoregressive step runs entirely on chip: FP32 RMSNorm/conv/recurrence
//     plus a W8A8 INT8 GEMV engine that streams packed Q8 weights from one HP
//     port and keeps the recurrent state resident in URAM/BRAM.
//   * otherwise            -> the CPU reference (USE_CPU_ONLY) that mirrors
//     references/gdn.c/runq.c bit-for-bit, and the FPGA host glue.
//
// W8A8 semantics follow runq.c: weights are symmetric int8 with one fp32 scale
// per 32-lane group; activations are dynamically quantized at each linear
// boundary; INT8 dot products accumulate in int32 and rescale to fp32.

#ifdef BUILD_DECODE_KERNEL

#include "decode.hpp"

#include <ap_int.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace gdn {
namespace {

constexpr int kAccStages = 8;

using ParameterBeat = ap_uint<128>;
using ParameterWord = ap_uint<512>;
using Q8ActivationBuffer = int8_t[kHiddenDim];
using Q8ScaleBuffer = float[kHiddenGroups];
using ConvState = float[kNumLayers][kKeyDim][kConvSize];
using RecurrentState = float[kNumLayers][kNumHeads][kStateBanks][kHeadKDim]
                            [kStateGroups];

static inline float bits_to_float(uint32_t bits) {
  union {
    uint32_t bits;
    float value;
  } converter;
  converter.bits = bits;
  return converter.value;
}

static inline float unpack_scale(const ParameterWord& word, int row) {
  return bits_to_float(word.range((row + 1) * 32 - 1, row * 32).to_uint());
}

// A pipelined FP32 adder: a single-cycle fadd is ~13 ns on this part, so every
// reduction is spread over kAccStages rotating partial sums and only collapsed
// at the end. This is what lets the reduction loops keep II=1 at 150 MHz.
static inline float fadd(float a, float b) {
  float y = a + b;
#pragma HLS BIND_OP variable = y op = fadd impl = fulldsp latency = 6
  return y;
}

static inline void clear_acc(float acc[kAccStages]) {
  for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
    acc[stage] = 0.0f;
  }
}

static inline float reduce_acc(const float acc[kAccStages]) {
  const float sum01 = fadd(acc[0], acc[1]);
  const float sum23 = fadd(acc[2], acc[3]);
  const float sum45 = fadd(acc[4], acc[5]);
  const float sum67 = fadd(acc[6], acc[7]);
  return fadd(fadd(sum01, sum23), fadd(sum45, sum67));
}

static inline float siluf(float x);

// ---------------------------------------------------------------------------
// One streaming Q8 engine for every projection, paired W1/W3 and LM argmax.
// A logical 512-bit word arrives as four 128-bit HP0 beats; arithmetic is done
// in the same II=4 loop, so there is no ParameterTile or score spill buffer.
// ---------------------------------------------------------------------------
static uint32_t q8_linear(float* output, const ParameterBeat* params,
                          int word_offset, int rows, int group_count,
                          const Q8ActivationBuffer& activation,
                          const Q8ScaleBuffer& scales, bool paired,
                          bool argmax_mode) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = activation cyclic factor = 32 dim = 1
#pragma HLS ARRAY_PARTITION variable = scales complete dim = 1

  float acc0[kPackedRows];
  float acc1[kPackedRows];
  int8_t input_group[kQuantGroupSize];
  ParameterWord scale_word = 0;
  float input_scale = 0.0f;
#pragma HLS ARRAY_PARTITION variable = acc0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = acc1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = input_group complete dim = 1

  float best_score[kAccStages];
  uint32_t best_row[kAccStages];
#pragma HLS ARRAY_PARTITION variable = best_score complete dim = 1
#pragma HLS ARRAY_PARTITION variable = best_row complete dim = 1
  for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
    best_score[stage] = -FLT_MAX;
    best_row[stage] = 0;
  }

  const int words = rows / kPackedRows * group_count * kWordsPerGroup *
                    (paired ? 2 : 1);
  int block = 0;
  int group = 0;
  int matrix = 0;
  int word_in_group = 0;
  const int last_group = group_count - 1;

  for (int it = 0; it < words; ++it) {
#pragma HLS PIPELINE II = 4
#pragma HLS DEPENDENCE variable = acc0 inter false
#pragma HLS DEPENDENCE variable = acc1 inter false
    const int beat = (word_offset + it) * kBeatsPerWord;
    ParameterWord word;
    word.range(127, 0) = params[beat + 0];
    word.range(255, 128) = params[beat + 1];
    word.range(383, 256) = params[beat + 2];
    word.range(511, 384) = params[beat + 3];

    if (word_in_group == 0) {
      scale_word = word;
      input_scale = scales[group];
      for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS UNROLL
        input_group[lane] = activation[group * kQuantGroupSize + lane];
      }
    } else {
      const int pair = word_in_group - 1;
      const int row0 = pair * 2;
      const int row1 = row0 + 1;
      int32_t dot0 = 0;
      int32_t dot1 = 0;
      for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS UNROLL
        const ap_int<8> w0 = word.range((lane + 1) * 8 - 1, lane * 8);
        const ap_int<8> w1 =
            word.range((kQuantGroupSize + lane + 1) * 8 - 1,
                       (kQuantGroupSize + lane) * 8);
        dot0 += w0.to_int() * static_cast<int32_t>(input_group[lane]);
        dot1 += w1.to_int() * static_cast<int32_t>(input_group[lane]);
      }
      const float partial0 =
          static_cast<float>(dot0) * input_scale * unpack_scale(scale_word, row0);
      const float partial1 =
          static_cast<float>(dot1) * input_scale * unpack_scale(scale_word, row1);
      const float prev0 = group == 0 ? 0.0f
                                     : (matrix == 0 ? acc0[row0] : acc1[row0]);
      const float prev1 = group == 0 ? 0.0f
                                     : (matrix == 0 ? acc0[row1] : acc1[row1]);
      const float sum0 = fadd(prev0, partial0);
      const float sum1 = fadd(prev1, partial1);
      if (matrix == 0) {
        acc0[row0] = sum0;
        acc0[row1] = sum1;
      } else {
        acc1[row0] = sum0;
        acc1[row1] = sum1;
      }

      if (group == last_group && (!paired || matrix == 1)) {
        const int output_row0 = block * kPackedRows + row0;
        const int output_row1 = output_row0 + 1;
        const float value0 = paired ? siluf(acc0[row0]) * sum0 : sum0;
        const float value1 = paired ? siluf(acc0[row1]) * sum1 : sum1;
        if (argmax_mode) {
          const int stage0 = output_row0 & (kAccStages - 1);
          const int stage1 = output_row1 & (kAccStages - 1);
          if (value0 > best_score[stage0]) {
            best_score[stage0] = value0;
            best_row[stage0] = output_row0;
          }
          if (value1 > best_score[stage1]) {
            best_score[stage1] = value1;
            best_row[stage1] = output_row1;
          }
        } else {
          output[output_row0] = value0;
          output[output_row1] = value1;
        }
      }
    }

    if (++word_in_group == kWordsPerGroup) {
      word_in_group = 0;
      if (paired && matrix == 0) {
        matrix = 1;
      } else if (++group == group_count) {
        group = 0;
        matrix = 0;
        ++block;
      } else {
        matrix = 0;
      }
    }
  }

  float final_score = best_score[0];
  uint32_t final_row = best_row[0];
  for (int stage = 1; stage < kAccStages; ++stage) {
    const bool better = best_score[stage] > final_score ||
                        (best_score[stage] == final_score &&
                         best_row[stage] < final_row);
    if (better) {
      final_score = best_score[stage];
      final_row = best_row[stage];
    }
  }
  return final_row;
}

// ---------------------------------------------------------------------------
// Controller-side primitives (FP32).
// ---------------------------------------------------------------------------
static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float siluf(float x) { return x * sigmoidf(x); }
static inline float softplusf(float x) {
  return x > 20.0f ? x : log1pf(expf(x));
}

template <int Size>
static void quantize(Q8ActivationBuffer& out, Q8ScaleBuffer& scales,
                     const float in[Size]) {
  for (int group = 0; group < Size / kQuantGroupSize; ++group) {
    float lane_max[kAccStages];
#pragma HLS ARRAY_PARTITION variable = lane_max complete dim = 1
    for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
      lane_max[stage] = 0.0f;
    }
    for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = lane_max inter false
      const float value = fabsf(in[group * kQuantGroupSize + lane]);
      const int stage = lane & (kAccStages - 1);
      if (value > lane_max[stage]) {
        lane_max[stage] = value;
      }
    }
    float max_abs = 0.0f;
    for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
      if (lane_max[stage] > max_abs) {
        max_abs = lane_max[stage];
      }
    }
    const float scale = max_abs / 127.0f;
    scales[group] = scale == 0.0f ? 1.0f : scale;
    for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS PIPELINE II = 1
      const int index = group * kQuantGroupSize + lane;
      float value = scale == 0.0f ? 0.0f : nearbyintf(in[index] / scale);
      if (value > 127.0f) {
        value = 127.0f;
      } else if (value < -127.0f) {
        value = -127.0f;
      }
      out[index] = static_cast<int8_t>(value);
    }
  }
}

template <int Size>
static void rmsnorm(float out[Size], const float in[Size],
                    const float* weight) {
  float acc[kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
  clear_acc(acc);
  for (int i = 0; i < Size; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
    const int stage = i & (kAccStages - 1);
    acc[stage] = fadd(acc[stage], in[i] * in[i]);
  }
  const float norm = 1.0f / sqrtf(reduce_acc(acc) / Size + 1e-5f);
  for (int i = 0; i < Size; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] = weight[i] * (in[i] * norm);
  }
}

static float dot(const float* a, const float* b, int size) {
  float acc[kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
  clear_acc(acc);
  for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
    const int stage = i & (kAccStages - 1);
    acc[stage] = fadd(acc[stage], a[i] * b[i]);
  }
  return reduce_acc(acc);
}

// conv_size is 4, so the delay line shifts and the tap dot product both fully
// unroll into a 2-level adder tree; one channel per cycle.
static void conv_step(float* channels, float state[][kConvSize],
                      const float* weight, int count, bool valid) {
  // Splitting the tap dimension gives the shift register one memory per tap, so
  // the whole delay line updates in a single cycle.
#pragma HLS ARRAY_PARTITION variable = state complete dim = 2
  for (int c = 0; c < count; ++c) {
#pragma HLS PIPELINE II = 1
    float taps[kConvSize];
#pragma HLS ARRAY_PARTITION variable = taps complete dim = 1
    for (int j = 0; j < kConvSize - 1; ++j) {
#pragma HLS UNROLL
      taps[j] = valid ? state[c][j + 1] : 0.0f;
    }
    taps[kConvSize - 1] = channels[c];

    float products[kConvSize];
#pragma HLS ARRAY_PARTITION variable = products complete dim = 1
    for (int j = 0; j < kConvSize; ++j) {
#pragma HLS UNROLL
      state[c][j] = taps[j];
      products[j] = taps[j] * weight[c * kConvSize + j];
    }
    const float sum =
        fadd(fadd(products[0], products[1]), fadd(products[2], products[3]));
    channels[c] = siluf(sum);
  }
}

// Sixteen value-column banks turn the 32x32 state into two 64-cycle sweeps.
static void recurrence_head(
    float S[kStateBanks][kHeadKDim][kStateGroups], const float* q,
    const float* k, const float* v, float beta, float decay, bool valid,
    float* out) {
#pragma HLS INLINE off
  float prediction[kHeadVDim];
  float delta[kHeadVDim];
#pragma HLS ARRAY_PARTITION variable = S complete dim = 1
#pragma HLS ARRAY_PARTITION variable = prediction complete dim = 1
#pragma HLS ARRAY_PARTITION variable = delta complete dim = 1
#pragma HLS ARRAY_PARTITION variable = out complete dim = 1

  for (int group = 0; group < kStateGroups; ++group) {
#pragma HLS LOOP_FLATTEN off
    float acc[kStateBanks][kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
    for (int bank = 0; bank < kStateBanks; ++bank) {
#pragma HLS UNROLL
      for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
        acc[bank][stage] = 0.0f;
      }
    }
    for (int i = 0; i < kHeadKDim; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
      const int stage = i & (kAccStages - 1);
      for (int bank = 0; bank < kStateBanks; ++bank) {
#pragma HLS UNROLL
        const float decayed = (valid ? S[bank][i][group] : 0.0f) * decay;
        S[bank][i][group] = decayed;
        acc[bank][stage] = fadd(acc[bank][stage], decayed * k[i]);
      }
    }
    for (int bank = 0; bank < kStateBanks; ++bank) {
#pragma HLS UNROLL
      prediction[group * kStateBanks + bank] = reduce_acc(acc[bank]);
    }
  }
  for (int j = 0; j < kHeadVDim; ++j) {
#pragma HLS PIPELINE II = 1
    delta[j] = (v[j] - prediction[j]) * beta;
  }

  for (int group = 0; group < kStateGroups; ++group) {
#pragma HLS LOOP_FLATTEN off
    float acc[kStateBanks][kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
    for (int bank = 0; bank < kStateBanks; ++bank) {
#pragma HLS UNROLL
      for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
        acc[bank][stage] = 0.0f;
      }
    }
    for (int i = 0; i < kHeadKDim; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
      const int stage = i & (kAccStages - 1);
      for (int bank = 0; bank < kStateBanks; ++bank) {
#pragma HLS UNROLL
        const int j = group * kStateBanks + bank;
        const float updated = fadd(S[bank][i][group], k[i] * delta[j]);
        S[bank][i][group] = updated;
        acc[bank][stage] = fadd(acc[bank][stage], q[i] * updated);
      }
    }
    for (int bank = 0; bank < kStateBanks; ++bank) {
#pragma HLS UNROLL
      out[group * kStateBanks + bank] = reduce_acc(acc[bank]);
    }
  }
}

static void head_rmsnorm_gated(float* x, const float* gate,
                               const float* weight) {
  float acc[kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
  clear_acc(acc);
  for (int i = 0; i < kHeadVDim; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
    const int stage = i & (kAccStages - 1);
    acc[stage] = fadd(acc[stage], x[i] * x[i]);
  }
  const float norm = 1.0f / sqrtf(reduce_acc(acc) / kHeadVDim + 1e-5f);
  for (int i = 0; i < kHeadVDim; ++i) {
#pragma HLS PIPELINE II = 1
    x[i] = weight[i] * (x[i] * norm) * siluf(gate[i]);
  }
}

static ParameterWord read_word(const ParameterBeat* params, int offset) {
#pragma HLS INLINE
  const int beat = offset * kBeatsPerWord;
  ParameterWord word;
  word.range(127, 0) = params[beat + 0];
  word.range(255, 128) = params[beat + 1];
  word.range(383, 256) = params[beat + 2];
  word.range(511, 384) = params[beat + 3];
  return word;
}

static void load_embedding(float x[kDim], const ParameterBeat* params,
                           int token) {
  const int row = token % kPackedRows;
  const int pair = row / 2;
  const int half = row & 1;
  const int base = kPackedTokOffset +
                   token / kPackedRows * kDimGroups * kWordsPerGroup;
  for (int group = 0; group < kDimGroups; ++group) {
    const ParameterWord scale_word =
        read_word(params, base + group * kWordsPerGroup);
    const ParameterWord weight_word =
        read_word(params, base + group * kWordsPerGroup + 1 + pair);
    const float scale = unpack_scale(scale_word, row);
    for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS PIPELINE II = 1
      const int bit = (half * kQuantGroupSize + lane) * 8;
      const ap_int<8> value = weight_word.range(bit + 7, bit);
      x[group * kQuantGroupSize + lane] = value.to_int() * scale;
    }
  }
}

static void forward(int token, int reset_state, const float* side,
                    const ParameterBeat* packed_params,
                    ConvState& q_conv_state, ConvState& k_conv_state,
                    ConvState& v_conv_state, RecurrentState& S,
                    bool valid[kNumLayers][kNumHeads], uint32_t* next_token) {
#pragma HLS ALLOCATION function instances = q8_linear limit = 1
  if (reset_state) {
    for (int layer = 0; layer < kNumLayers; ++layer) {
      for (int head = 0; head < kNumHeads; ++head) {
#pragma HLS UNROLL
        valid[layer][head] = false;
      }
    }
  }

  float x[kDim];
  float attn_norm[kDim];
  float q[kHeadKDim];
  float k[kHeadKDim];
  float v[kHeadVDim];
  float gate[kHeadVDim];
  float head_out[kHeadVDim];
#pragma HLS ARRAY_PARTITION variable = head_out complete dim = 1
  float linear_out[kValueDim];
  float ffn_norm[kDim];
  float q8_out[kHiddenDim];
  Q8ActivationBuffer act_q8;
  Q8ScaleBuffer act_scales;
  volatile uint32_t q8_result = 0;
  volatile int q8_rows = 0;
  volatile int q8_groups = 0;
  volatile bool q8_paired = false;
  volatile bool q8_argmax = false;

  load_embedding(x, packed_params, token);

  const float q_scale = 1.0f / sqrtf(static_cast<float>(kHeadKDim));

  for (int layer = 0; layer < kNumLayers; ++layer) {
    const float* rms_att = side + SideRmsAttOffset(layer);
    const float* rms_ffn = side + SideRmsFfnOffset(layer);
    const float* a_proj = side + SideAProjOffset(layer);
    const float* b_proj = side + SideBProjOffset(layer);
    const float* q_conv = side + SideQConvOffset(layer);
    const float* k_conv = side + SideKConvOffset(layer);
    const float* v_conv = side + SideVConvOffset(layer);
    const float* a_decay = side + SideAOffset(layer);
    const float* dt_bias = side + SideDtBiasOffset(layer);
    const float* o_norm = side + SideONormOffset(layer);

    rmsnorm<kDim>(attn_norm, x, rms_att);
    quantize<kDim>(act_q8, act_scales, attn_norm);
    for (int h = 0; h < kNumHeads; ++h) {
      q8_rows = 2 * (kHeadKDim + kHeadVDim);
      q8_groups = kDimGroups;
      q8_paired = false;
      q8_argmax = false;
      q8_result = q8_linear(q8_out, packed_params, PackedQOffset(layer, h),
                            q8_rows, q8_groups, act_q8, act_scales, q8_paired,
                            q8_argmax);
      for (int i = 0; i < kHeadKDim; ++i) {
#pragma HLS PIPELINE II = 1
        q[i] = q8_out[i];
        k[i] = q8_out[kHeadKDim + i];
      }
      for (int i = 0; i < kHeadVDim; ++i) {
#pragma HLS PIPELINE II = 1
        v[i] = q8_out[2 * kHeadKDim + i];
        gate[i] = q8_out[2 * kHeadKDim + kHeadVDim + i];
      }

      conv_step(q, q_conv_state[layer] + h * kHeadKDim,
                q_conv + h * kHeadKDim * kConvSize, kHeadKDim,
                valid[layer][h]);
      conv_step(k, k_conv_state[layer] + h * kHeadKDim,
                k_conv + h * kHeadKDim * kConvSize, kHeadKDim,
                valid[layer][h]);
      conv_step(v, v_conv_state[layer] + h * kHeadVDim,
                v_conv + h * kHeadVDim * kConvSize, kHeadVDim,
                valid[layer][h]);

      float qacc[kAccStages];
      float kacc[kAccStages];
#pragma HLS ARRAY_PARTITION variable = qacc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = kacc complete dim = 1
      clear_acc(qacc);
      clear_acc(kacc);
      for (int i = 0; i < kHeadKDim; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = qacc inter false
#pragma HLS DEPENDENCE variable = kacc inter false
        const int stage = i & (kAccStages - 1);
        qacc[stage] = fadd(qacc[stage], q[i] * q[i]);
        kacc[stage] = fadd(kacc[stage], k[i] * k[i]);
      }
      const float qn = 1.0f / sqrtf(reduce_acc(qacc) + 1e-6f);
      const float kn = 1.0f / sqrtf(reduce_acc(kacc) + 1e-6f);
      for (int i = 0; i < kHeadKDim; ++i) {
#pragma HLS PIPELINE II = 1
        q[i] = q[i] * qn * q_scale;
        k[i] = k[i] * kn;
      }
      const float beta = sigmoidf(dot(attn_norm, b_proj + h * kDim, kDim));
      const float decay = expf(
          a_decay[h] *
          softplusf(dot(attn_norm, a_proj + h * kDim, kDim) + dt_bias[h]));
      recurrence_head(S[layer][h], q, k, v, beta, decay, valid[layer][h],
                      head_out);
      head_rmsnorm_gated(head_out, gate, o_norm);
      for (int i = 0; i < kHeadVDim; ++i) {
#pragma HLS PIPELINE II = 1
        linear_out[h * kHeadVDim + i] = head_out[i];
      }
      valid[layer][h] = true;
    }

    quantize<kValueDim>(act_q8, act_scales, linear_out);
    q8_rows = kDim;
    q8_groups = kValueGroups;
    q8_result = q8_linear(q8_out, packed_params, PackedOOffset(layer), q8_rows,
                          q8_groups, act_q8, act_scales, q8_paired,
                          q8_argmax);
    for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
      x[i] += q8_out[i];
    }

    rmsnorm<kDim>(ffn_norm, x, rms_ffn);
    quantize<kDim>(act_q8, act_scales, ffn_norm);
    q8_rows = kHiddenDim;
    q8_groups = kDimGroups;
    q8_paired = true;
    q8_result = q8_linear(q8_out, packed_params, PackedW13Offset(layer),
                          q8_rows, q8_groups, act_q8, act_scales, q8_paired,
                          q8_argmax);
    quantize<kHiddenDim>(act_q8, act_scales, q8_out);
    q8_rows = kDim;
    q8_groups = kHiddenGroups;
    q8_paired = false;
    q8_result = q8_linear(q8_out, packed_params, PackedW2Offset(layer),
                          q8_rows, q8_groups, act_q8, act_scales, q8_paired,
                          q8_argmax);
    for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
      x[i] += q8_out[i];
    }
  }

  rmsnorm<kDim>(attn_norm, x, side + kSideRmsFinalOffset);
  quantize<kDim>(act_q8, act_scales, attn_norm);
  q8_rows = kVocabSize;
  q8_groups = kDimGroups;
  q8_argmax = true;
  q8_result = q8_linear(q8_out, packed_params, kPackedTokOffset, q8_rows,
                        q8_groups, act_q8, act_scales, q8_paired, q8_argmax);
  *next_token = q8_result;
}

} // namespace
} // namespace gdn

using namespace gdn;

extern "C" {

void decode(int token, int reset_state,
            const ParameterBeat* __restrict packed_params, const float* side,
            uint32_t* next_token) {
#pragma HLS INTERFACE m_axi port = packed_params bundle = params0 \
    max_read_burst_length = 256 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = side bundle = gmem \
    max_read_burst_length = 64 num_read_outstanding = 4
#pragma HLS INTERFACE m_axi port = next_token bundle = gmem \
    num_write_outstanding = 2
#pragma HLS INTERFACE s_axilite port = token bundle = control
#pragma HLS INTERFACE s_axilite port = reset_state bundle = control
#pragma HLS INTERFACE s_axilite port = packed_params bundle = control
#pragma HLS INTERFACE s_axilite port = side bundle = control
#pragma HLS INTERFACE s_axilite port = next_token bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  static ConvState q_conv_state;
  static ConvState k_conv_state;
  static ConvState v_conv_state;
  static RecurrentState S;
  static bool valid[kNumLayers][kNumHeads];
#pragma HLS BIND_STORAGE variable = q_conv_state type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = k_conv_state type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = v_conv_state type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = S type = ram_2p impl = uram
  // The conv delay line updates all taps of a channel in one cycle, so the tap
  // dimension has to be split at the declaration too, not just in conv_step.
#pragma HLS ARRAY_PARTITION variable = q_conv_state complete dim = 3
#pragma HLS ARRAY_PARTITION variable = k_conv_state complete dim = 3
#pragma HLS ARRAY_PARTITION variable = v_conv_state complete dim = 3
#pragma HLS ARRAY_PARTITION variable = S complete dim = 3
#pragma HLS ARRAY_PARTITION variable = valid complete dim = 0

  forward(token, reset_state, side, packed_params, q_conv_state, k_conv_state,
          v_conv_state, S, valid, next_token);
}

} // extern "C"

#else // !BUILD_DECODE_KERNEL

#include "decode.hpp"
#include "weight.hpp"

#include <cmath>
#include <cstdint>

namespace gdn {

RunState::RunState()
    : x(kDim), xb(kDim > kValueDim ? kDim : kValueDim), hb(kHiddenDim),
      hb2(kHiddenDim), q(kKeyDim), k(kKeyDim), v(kValueDim),
      gate(kValueDim), beta(kNumHeads), decay(kNumHeads),
      linear_out(kValueDim), q_conv_state(kQConvStateCount),
      k_conv_state(kKConvStateCount), v_conv_state(kVConvStateCount),
      S(kSStateCount), xq_q(kHiddenDim), xq_s(kHiddenGroups) {}

namespace {

void Quantize(std::int8_t* q, float* s, const float* x, int n) {
  const int groups = n / kQuantGroupSize;
  for (int group = 0; group < groups; ++group) {
    const int offset = group * kQuantGroupSize;
    float wmax = 0.0f;
    for (int i = 0; i < kQuantGroupSize; ++i) {
      const float value = std::fabs(x[offset + i]);
      if (value > wmax) {
        wmax = value;
      }
    }
    if (wmax == 0.0f) {
      s[group] = 1.0f;
      for (int i = 0; i < kQuantGroupSize; ++i) {
        q[offset + i] = 0;
      }
      continue;
    }
    const float scale = wmax / 127.0f;
    s[group] = scale;
    for (int i = 0; i < kQuantGroupSize; ++i) {
      int quantized = static_cast<int>(std::lround(x[offset + i] / scale));
      if (quantized > 127) quantized = 127;
      if (quantized < -127) quantized = -127;
      q[offset + i] = static_cast<std::int8_t>(quantized);
    }
  }
}

void Matmul(float* out, const std::int8_t* xq, const float* xs,
            const QuantizedTensor& w) {
  const int n = w.cols;
  const int d = w.rows;
  const int groups = w.groups;
  for (int i = 0; i < d; ++i) {
    float val = 0.0f;
    const std::size_t q_row = static_cast<std::size_t>(i) * n;
    const std::size_t s_row = static_cast<std::size_t>(i) * groups;
    for (int group = 0; group < groups; ++group) {
      std::int32_t ival = 0;
      const int offset = group * kQuantGroupSize;
      for (int j = 0; j < kQuantGroupSize; ++j) {
        ival += static_cast<std::int32_t>(xq[offset + j]) *
                static_cast<std::int32_t>(w.q[q_row + offset + j]);
      }
      val += static_cast<float>(ival) * w.s[s_row + group] * xs[group];
    }
    out[i] = val;
  }
}

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float Silu(float x) { return x * Sigmoid(x); }
float Softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

void RmsNorm(float* o, const float* x, const float* weight, int size) {
  float ss = 0.0f;
  for (int i = 0; i < size; ++i) ss += x[i] * x[i];
  ss = 1.0f / std::sqrt(ss / size + 1e-5f);
  for (int i = 0; i < size; ++i) o[i] = weight[i] * (ss * x[i]);
}

void L2Norm(float* x, int size) {
  float ss = 0.0f;
  for (int i = 0; i < size; ++i) ss += x[i] * x[i];
  ss = 1.0f / std::sqrt(ss + 1e-6f);
  for (int i = 0; i < size; ++i) x[i] *= ss;
}

float DotScalar(const float* a, const float* b, int n) {
  float val = 0.0f;
  for (int i = 0; i < n; ++i) val += a[i] * b[i];
  return val;
}

void ConvStep(float* channels, float* state, const float* weight, int count) {
  for (int c = 0; c < count; ++c) {
    float* st = state + c * kConvSize;
    for (int i = 0; i < kConvSize - 1; ++i) st[i] = st[i + 1];
    st[kConvSize - 1] = channels[c];
    float value = 0.0f;
    for (int i = 0; i < kConvSize; ++i) value += st[i] * weight[c * kConvSize + i];
    channels[c] = Silu(value);
  }
}

} // namespace

void CpuForward(RunState& s, const Weights& w, int token, float* logits) {
  const std::size_t q_base = static_cast<std::size_t>(token) * kDim;
  const std::size_t s_base = static_cast<std::size_t>(token) * kDimGroups;
  for (int i = 0; i < kDim; ++i) {
    s.x[i] = static_cast<float>(w.tok_emb.q[q_base + i]) *
             w.tok_emb.s[s_base + i / kQuantGroupSize];
  }

  const float q_scale = 1.0f / std::sqrt(static_cast<float>(kHeadKDim));

  for (int layer = 0; layer < kNumLayers; ++layer) {
    const LayerWeights& l = w.layers[layer];

    // Mixer.
    RmsNorm(s.xb.data(), s.x.data(), l.attn_norm.data(), kDim);
    Quantize(s.xq_q.data(), s.xq_s.data(), s.xb.data(), kDim);
    Matmul(s.q.data(), s.xq_q.data(), s.xq_s.data(), l.q_proj);
    Matmul(s.k.data(), s.xq_q.data(), s.xq_s.data(), l.k_proj);
    Matmul(s.v.data(), s.xq_q.data(), s.xq_s.data(), l.v_proj);
    Matmul(s.gate.data(), s.xq_q.data(), s.xq_s.data(), l.g_proj);

    ConvStep(s.q.data(),
             s.q_conv_state.data() + static_cast<std::size_t>(layer) * kQConvSize,
             l.q_conv.data(), kKeyDim);
    ConvStep(s.k.data(),
             s.k_conv_state.data() + static_cast<std::size_t>(layer) * kKConvSize,
             l.k_conv.data(), kKeyDim);
    ConvStep(s.v.data(),
             s.v_conv_state.data() + static_cast<std::size_t>(layer) * kVConvSize,
             l.v_conv.data(), kValueDim);

    for (int h = 0; h < kNumHeads; ++h) {
      float* qh = s.q.data() + h * kHeadKDim;
      float* kh = s.k.data() + h * kHeadKDim;
      L2Norm(qh, kHeadKDim);
      L2Norm(kh, kHeadKDim);
      for (int i = 0; i < kHeadKDim; ++i) qh[i] *= q_scale;
      s.beta[h] = Sigmoid(DotScalar(s.xb.data(), l.b_proj.data() + h * kDim, kDim));
      s.decay[h] =
          std::exp(l.A[h] * Softplus(DotScalar(s.xb.data(),
                                                l.a_proj.data() + h * kDim, kDim) +
                                     l.dt_bias[h]));
    }

    float* S_layer =
        s.S.data() + static_cast<std::size_t>(layer) * kNumHeads * kHeadKDim *
                         kHeadVDim;
    for (int h = 0; h < kNumHeads; ++h) {
      float* S = S_layer + static_cast<std::size_t>(h) * kHeadKDim * kHeadVDim;
      const float* qh = s.q.data() + h * kHeadKDim;
      const float* kh = s.k.data() + h * kHeadKDim;
      const float* vh = s.v.data() + h * kHeadVDim;
      float* out = s.linear_out.data() + h * kHeadVDim;

      for (int i = 0; i < kHeadKDim * kHeadVDim; ++i) S[i] *= s.decay[h];
      for (int j = 0; j < kHeadVDim; ++j) {
        float prediction = 0.0f;
        for (int i = 0; i < kHeadKDim; ++i) prediction += S[i * kHeadVDim + j] * kh[i];
        s.xb[j] = (vh[j] - prediction) * s.beta[h];
      }
      for (int i = 0; i < kHeadKDim; ++i) {
        for (int j = 0; j < kHeadVDim; ++j) S[i * kHeadVDim + j] += kh[i] * s.xb[j];
      }
      for (int j = 0; j < kHeadVDim; ++j) {
        float value = 0.0f;
        for (int i = 0; i < kHeadKDim; ++i) value += qh[i] * S[i * kHeadVDim + j];
        out[j] = value;
      }
    }

    for (int h = 0; h < kNumHeads; ++h) {
      float* xh = s.linear_out.data() + h * kHeadVDim;
      const float* gh = s.gate.data() + h * kHeadVDim;
      float ss = 0.0f;
      for (int i = 0; i < kHeadVDim; ++i) ss += xh[i] * xh[i];
      ss = 1.0f / std::sqrt(ss / kHeadVDim + 1e-5f);
      for (int i = 0; i < kHeadVDim; ++i)
        xh[i] = l.o_norm[i] * (ss * xh[i]) * Silu(gh[i]);
    }
    Quantize(s.xq_q.data(), s.xq_s.data(), s.linear_out.data(), kValueDim);
    Matmul(s.xb.data(), s.xq_q.data(), s.xq_s.data(), l.o_proj);
    for (int i = 0; i < kDim; ++i) s.x[i] += s.xb[i];

    // FFN.
    RmsNorm(s.xb.data(), s.x.data(), l.ffn_norm.data(), kDim);
    Quantize(s.xq_q.data(), s.xq_s.data(), s.xb.data(), kDim);
    Matmul(s.hb.data(), s.xq_q.data(), s.xq_s.data(), l.w1);
    Matmul(s.hb2.data(), s.xq_q.data(), s.xq_s.data(), l.w3);
    for (int i = 0; i < kHiddenDim; ++i) s.hb[i] = Silu(s.hb[i]) * s.hb2[i];
    Quantize(s.xq_q.data(), s.xq_s.data(), s.hb.data(), kHiddenDim);
    Matmul(s.xb.data(), s.xq_q.data(), s.xq_s.data(), l.w2);
    for (int i = 0; i < kDim; ++i) s.x[i] += s.xb[i];
  }

  RmsNorm(s.x.data(), s.x.data(), w.rms_final.data(), kDim);
  Quantize(s.xq_q.data(), s.xq_s.data(), s.x.data(), kDim);
  Matmul(logits, s.xq_q.data(), s.xq_s.data(), w.tok_emb);
}

int ArgmaxLogits(const float* logits, int count) {
  int best = 0;
  float best_score = logits[0];
  for (int i = 1; i < count; ++i) {
    if (logits[i] > best_score) {
      best_score = logits[i];
      best = i;
    }
  }
  return best;
}

} // namespace gdn

#if !defined(USE_CPU_ONLY)
namespace gdn {

int Decode(int token, bool reset_state, cl::CommandQueue& q, cl::Kernel& kernel,
           std::uint32_t* next_token, cl::Buffer& next_token_buffer) {
  cl_int err = CL_SUCCESS;
  err = kernel.setArg(0, token);
  if (err != CL_SUCCESS) return -1;
  err = kernel.setArg(1, reset_state ? 1 : 0);
  if (err != CL_SUCCESS) return -1;
  err = q.enqueueTask(kernel);
  if (err != CL_SUCCESS) return -1;
  err = q.enqueueMigrateMemObjects({next_token_buffer},
                                   CL_MIGRATE_MEM_OBJECT_HOST);
  if (err != CL_SUCCESS) return -1;
  err = q.finish();
  if (err != CL_SUCCESS) return -1;
  return static_cast<int>(*next_token);
}

} // namespace gdn
#endif // !USE_CPU_ONLY

#endif // BUILD_DECODE_KERNEL
