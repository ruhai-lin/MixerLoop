// Gated DeltaNet 15M single-kernel decoder.
//
// Two builds share this file:
//   * BUILD_DECODE_KERNEL  -> structural HLS kernel. One canonical DATAFLOW
//     region; each physical module is a single long-running process.
//   * otherwise            -> CPU Q8 reference and FPGA host glue. That half
//     is the numerical oracle and must not change.
//
// Frozen process graph (do not add compute call sites or split these):
//   schedule_process, memory_process, scratch_process, weight_router,
//   q8_process, beta_process, conv_process, rec_process, post_process.
//
// Rules: one Q8 call site, one HP0 owner (memory_process), each stream
// SPSC. Shared hardware is one process looping over commands, never many
// helper call sites. Weight-source mux sits in weight_router; Q8 only sees
// a unified weight_stream. Recurrence state stays inside rec_process.

#ifdef BUILD_DECODE_KERNEL

#include "decode.hpp"

#include <ap_int.h>
#include <hls_stream.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#ifndef __SYNTHESIS__
#include <thread>
#endif

namespace gdn {
namespace {

constexpr int kVecLanes = 16;
constexpr int kAccStages = 8;
constexpr int kHeadGroups = kHeadVDim / kVecLanes;
constexpr int kSWords =
    kMaxLoopCount * kNumLayers * kNumHeads * kHeadKDim * kHeadGroups;
constexpr int kConvKinds = 3;
constexpr int kScratchDepth = 12288;
constexpr int kScratchBanks = 8;
constexpr int kMixerCacheBase = 0;
// Frozen Mixer Q8 cache: scratch[0, 5759], 8×64-bit banks, 24 URAM.
// loop0 writes all 5760 words; loop1–3 read 512 bit/cycle from the same window.
constexpr int kMixerCacheWords = kPackedQKVGWords + kPackedProjWords;
constexpr int kFfnRingBase = kMixerCacheWords;
constexpr int kFfnRingWords = 6528;
constexpr int kFfnPackedWords = 2 * kPackedW13Words + kPackedW2Words;
constexpr int kPackedOHeadWords = PackedMatrixWords(kDim, 1);
constexpr int kQkvgRows = 2 * (kHeadKDim + kHeadVDim);
static_assert(kMixerCacheWords == 5760, "Mixer cache must be 5760 packed words");
static_assert(kScratchBanks == 8, "Mixer scratch is 8×64-bit");
static_assert(kMixerCacheBase == 0, "Mixer cache occupies scratch[0, 5759]");
static_assert(kFfnRingBase + kFfnRingWords == kScratchDepth,
              "FFN ring must occupy scratch[5760, 12287]");
static_assert(kPackedOHeadWords == 144, "one O input-head group is 144 words");
static_assert(2 * kPackedW13Words == 6912, "paired W1/W3 must be 6912 words");
static_assert(kPackedW2Words == 3456, "W2 must be 3456 words");

using ParameterBeat = ap_uint<128>;
using ParameterWord = ap_uint<512>;
using ConvWord = ap_uint<96>;
using ActGroup = ap_uint<256>;
using FloatPair = ap_uint<64>;
using StateBank = FloatPair;

struct ActTable {
  ActGroup group[kHiddenGroups];
  float scale[kHiddenGroups];
};

struct ActPkt {
  ActGroup group;
  float scale;
};

using ConvHist = ConvWord[kNumLayers][kMaxLoopCount][kConvKinds][kKeyDim];
using StateMem = StateBank[8][kSWords];
using ScratchMem = StateBank[kScratchBanks][kScratchDepth];

enum TokenPhase {
  kPhaseBegin = 0,
  kPhaseLayer,
  kPhaseLm,
  kPhaseEnd
};
enum WeightKind {
  kWeightNone = 0,
  kWeightMixerQkvg,
  kWeightMixerO,
  kWeightW13,
  kWeightW2,
  kWeightLm
};
enum MemKind {
  kMemEmbed = 0,
  kMemSide,
  kMemFill,
  kMemFfnPrefetch,
  kMemLm,
  kMemStore
};
enum ScratchKind {
  kScratchFill = 0,
  kScratchRead,
  kScratchORead,
  kScratchRingBegin,
  kScratchRingRead
};
enum Q8Kind { kQ8Qkvg = 0, kQ8O, kQ8W13, kQ8W2, kQ8Lm };
enum PostKind {
  kPostEmbed = 0,
  kPostAttn,
  kPostPackO,
  kPostResO,
  kPostFfn,
  kPostW13,
  kPostResW2,
  kPostFinal
};

struct CmdHeader {
  unsigned char phase;
  unsigned char layer_id;
  unsigned char loop_id;
  unsigned char weight_kind;
};

struct MemCmd {
  CmdHeader hdr;
  unsigned char kind;
  int token;
};
struct ScratchCmd {
  CmdHeader hdr;
  unsigned char kind;
};
struct RouteCmd {
  CmdHeader hdr;
};
struct Q8Cmd {
  CmdHeader hdr;
  unsigned char kind;
};
struct StageCmd {
  CmdHeader hdr;
};
struct PostCmd {
  CmdHeader hdr;
  unsigned char kind;
};

static CmdHeader make_hdr(unsigned char phase, unsigned char layer,
                          unsigned char loop, unsigned char weight_kind) {
#pragma HLS INLINE
  CmdHeader hdr;
  hdr.phase = phase;
  hdr.layer_id = layer;
  hdr.loop_id = loop;
  hdr.weight_kind = weight_kind;
  return hdr;
}

static int clamp_loop_count(int loop_count) {
#pragma HLS INLINE
  if (loop_count < 1) {
    return 1;
  }
  if (loop_count > kMaxLoopCount) {
    return kMaxLoopCount;
  }
  return loop_count;
}

static inline uint32_t float_to_bits(float value) {
  union {
    float value;
    uint32_t bits;
  } converter;
  converter.value = value;
  return converter.bits;
}

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

static inline float fadd(float a, float b) {
  float y = a + b;
#pragma HLS BIND_OP variable = y op = fadd impl = fulldsp latency = 6
  return y;
}

static inline float fsub(float a, float b) {
  float y = a - b;
#pragma HLS BIND_OP variable = y op = fsub impl = fulldsp latency = 6
  return y;
}

static inline float fmul(float a, float b) {
  float y = a * b;
#pragma HLS BIND_OP variable = y op = fmul impl = maxdsp latency = 4
  return y;
}

static inline void vec_clear(float acc[kVecLanes][kAccStages]) {
#pragma HLS INLINE
  for (int lane = 0; lane < kVecLanes; ++lane) {
#pragma HLS UNROLL
    for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
      acc[lane][stage] = 0.0f;
    }
  }
}

static inline float reduce8(const float acc[kAccStages]) {
#pragma HLS INLINE
  const float sum01 = fadd(acc[0], acc[1]);
  const float sum23 = fadd(acc[2], acc[3]);
  const float sum45 = fadd(acc[4], acc[5]);
  const float sum67 = fadd(acc[6], acc[7]);
  return fadd(fadd(sum01, sum23), fadd(sum45, sum67));
}

enum SfuOp { kSfuExp = 0, kSfuLog1p, kSfuSqrt, kSfuRecip };

static inline float sfu(int op, float x) {
#pragma HLS INLINE
  if (op == kSfuExp) {
    return expf(x);
  }
  if (op == kSfuLog1p) {
    return log1pf(x);
  }
  if (op == kSfuSqrt) {
    return sqrtf(x);
  }
  return 1.0f / x;
}

static inline float sfu_exp(float x) {
#pragma HLS INLINE
  return sfu(kSfuExp, x);
}
static inline float sfu_log1p(float x) {
#pragma HLS INLINE
  return sfu(kSfuLog1p, x);
}
static inline float sfu_sqrt(float x) {
#pragma HLS INLINE
  return sfu(kSfuSqrt, x);
}
static inline float sfu_recip(float x) {
#pragma HLS INLINE
  return sfu(kSfuRecip, x);
}
static inline float sfu_sigmoid(float x) {
#pragma HLS INLINE
  return sfu_recip(1.0f + sfu_exp(-x));
}
static inline float sfu_silu(float x) {
#pragma HLS INLINE
  return fmul(x, sfu_sigmoid(x));
}
static inline float sfu_softplus(float x) {
#pragma HLS INLINE
  return x > 20.0f ? x : sfu_log1p(sfu_exp(x));
}
static inline float sfu_rsqrt(float x) {
#pragma HLS INLINE
  return sfu_recip(sfu_sqrt(x));
}

static inline int state_addr(int loop, int layer, int head, int row, int group) {
#pragma HLS INLINE
  return (((loop * kNumLayers + layer) * kNumHeads + head) * kHeadKDim + row) *
         kHeadGroups +
         group;
}

static inline FloatPair pack2(float a, float b) {
#pragma HLS INLINE
  FloatPair word = 0;
  word.range(31, 0) = float_to_bits(a);
  word.range(63, 32) = float_to_bits(b);
  return word;
}

static inline void unpack2(FloatPair word, float& a, float& b) {
#pragma HLS INLINE
  a = bits_to_float(word.range(31, 0).to_uint());
  b = bits_to_float(word.range(63, 32).to_uint());
}

static inline void s_load(const StateMem banks, int addr, float lane[kVecLanes]) {
#pragma HLS INLINE
  for (int b = 0; b < 8; ++b) {
#pragma HLS UNROLL
    unpack2(banks[b][addr], lane[2 * b], lane[2 * b + 1]);
  }
}

static inline void s_store(StateMem banks, int addr, const float lane[kVecLanes]) {
#pragma HLS INLINE
  for (int b = 0; b < 8; ++b) {
#pragma HLS UNROLL
    banks[b][addr] = pack2(lane[2 * b], lane[2 * b + 1]);
  }
}

static inline ConvWord pack3(float a, float b, float c) {
#pragma HLS INLINE
  ConvWord word = 0;
  word.range(31, 0) = float_to_bits(a);
  word.range(63, 32) = float_to_bits(b);
  word.range(95, 64) = float_to_bits(c);
  return word;
}

static inline void unpack3(ConvWord word, float& a, float& b, float& c) {
#pragma HLS INLINE
  a = bits_to_float(word.range(31, 0).to_uint());
  b = bits_to_float(word.range(63, 32).to_uint());
  c = bits_to_float(word.range(95, 64).to_uint());
}

static inline ParameterWord read_word(const ParameterBeat* params, int offset) {
#pragma HLS INLINE
  const int beat = offset * kBeatsPerWord;
  ParameterWord word;
  word.range(127, 0) = params[beat + 0];
  word.range(255, 128) = params[beat + 1];
  word.range(383, 256) = params[beat + 2];
  word.range(511, 384) = params[beat + 3];
  return word;
}

static void pack_act_group(ActTable& act, int group,
                           const int8_t lanes[kQuantGroupSize], float scale) {
#pragma HLS INLINE
  ActGroup packed = 0;
  for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS UNROLL
    packed.range((lane + 1) * 8 - 1, lane * 8) =
        static_cast<uint8_t>(lanes[lane]);
  }
  act.group[group] = packed;
  act.scale[group] = scale;
}

static void unpack_act_group(const ActTable& act, int group,
                             int8_t lanes[kQuantGroupSize], float& scale) {
#pragma HLS INLINE
  const ActGroup packed = act.group[group];
  scale = act.scale[group];
  for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS UNROLL
    lanes[lane] = static_cast<int8_t>(
        packed.range((lane + 1) * 8 - 1, lane * 8).to_int());
  }
}

static int q8_word_count(int rows, int group_count, bool paired) {
#pragma HLS INLINE
  return rows / kPackedRows * group_count * kWordsPerGroup * (paired ? 2 : 1);
}

static ParameterWord scratch_read(const ScratchMem scratch, ap_uint<14> addr) {
#pragma HLS INLINE
  ParameterWord word;
  for (int bank = 0; bank < kScratchBanks; ++bank) {
#pragma HLS UNROLL
    word.range((bank + 1) * 64 - 1, bank * 64) = scratch[bank][addr];
  }
  return word;
}

static void scratch_write(ScratchMem scratch, ap_uint<14> addr,
                          ParameterWord word) {
#pragma HLS INLINE
  for (int bank = 0; bank < kScratchBanks; ++bank) {
#pragma HLS UNROLL
    scratch[bank][addr] = word.range((bank + 1) * 64 - 1, bank * 64);
  }
}

static void stream_write_vec(hls::stream<float>& dst, const float* src, int n) {
#pragma HLS INLINE
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
    dst.write(src[i]);
  }
}

static void stream_read_vec(hls::stream<float>& src, float* dst, int n) {
#pragma HLS INLINE
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
    dst[i] = src.read();
  }
}

static void stream_write_pairs(hls::stream<FloatPair>& dst, const float* src,
                               int n) {
#pragma HLS INLINE
  for (int i = 0; i < n; i += 2) {
#pragma HLS PIPELINE II = 1
    dst.write(pack2(src[i], src[i + 1]));
  }
}

static void stream_read_pairs(hls::stream<FloatPair>& src, float* dst, int n) {
#pragma HLS INLINE
  for (int i = 0; i < n; i += 2) {
#pragma HLS PIPELINE II = 1
    unpack2(src.read(), dst[i], dst[i + 1]);
  }
}

static void q8_consume(hls::stream<ParameterWord>& words, int count,
                       int group_count, const ActTable& act,
                       hls::stream<ActPkt>& acts, bool paired, bool argmax_mode,
                       bool head_o, unsigned char kind,
                       hls::stream<FloatPair>& qkvg,
                       hls::stream<FloatPair>& o_vec,
                       hls::stream<FloatPair>& w13_vec,
                       hls::stream<FloatPair>& w2_vec,
                       uint32_t* result) {
#pragma HLS INLINE

  float acc0[kPackedRows];
  float acc1[kPackedRows];
  float o_acc[kDim];
  int8_t input_group[kQuantGroupSize];
  ParameterWord scale_word = 0;
  float input_scale = 0.0f;
#pragma HLS ARRAY_PARTITION variable = acc0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = acc1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = o_acc cyclic factor = 2 dim = 1
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

  const int last_group = group_count - 1;
  const int heads = head_o ? kNumHeads : 1;

  for (int head = 0; head < heads; ++head) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 8
    if (head_o) {
      const ActPkt pkt = acts.read();
      input_scale = pkt.scale;
      for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS UNROLL
        input_group[lane] = static_cast<int8_t>(
            pkt.group.range((lane + 1) * 8 - 1, lane * 8).to_int());
      }
    }

    int block = 0;
    int group = 0;
    int matrix = 0;
    int word_in_group = 0;
    for (int it = 0; it < count; ++it) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc0 inter false
#pragma HLS DEPENDENCE variable = acc1 inter false
#pragma HLS DEPENDENCE variable = o_acc inter false
#pragma HLS DEPENDENCE variable = best_score inter false
#pragma HLS DEPENDENCE variable = best_row inter false
      const ParameterWord word = words.read();

      if (word_in_group == 0) {
        scale_word = word;
        if (!head_o) {
          unpack_act_group(act, group, input_group, input_scale);
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
        const float partial0 = static_cast<float>(dot0) * input_scale *
                               unpack_scale(scale_word, row0);
        const float partial1 = static_cast<float>(dot1) * input_scale *
                               unpack_scale(scale_word, row1);
        const float prev0 =
            group == 0 ? 0.0f : (matrix == 0 ? acc0[row0] : acc1[row0]);
        const float prev1 =
            group == 0 ? 0.0f : (matrix == 0 ? acc0[row1] : acc1[row1]);
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
          const float value0 =
              paired ? fmul(sfu_silu(acc0[row0]), sum0) : sum0;
          const float value1 =
              paired ? fmul(sfu_silu(acc0[row1]), sum1) : sum1;
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
          } else if (head_o) {
            const float old0 = head == 0 ? 0.0f : o_acc[output_row0];
            const float old1 = head == 0 ? 0.0f : o_acc[output_row1];
            const float total0 = fadd(old0, value0);
            const float total1 = fadd(old1, value1);
            o_acc[output_row0] = total0;
            o_acc[output_row1] = total1;
            if (head == kNumHeads - 1) {
              o_vec.write(pack2(total0, total1));
            }
          } else if (kind == kQ8Qkvg) {
            qkvg.write(pack2(value0, value1));
          } else if (kind == kQ8W13) {
            w13_vec.write(pack2(value0, value1));
          } else {
            w2_vec.write(pack2(value0, value1));
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
  if (argmax_mode) {
    *result = final_row;
  }
}

static void fp8_rmsnorm(float* out, const float* in, const float* weight,
                        int size) {
#pragma HLS INLINE
  float acc[kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
  for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
    acc[stage] = 0.0f;
  }
  for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
    const int stage = i & (kAccStages - 1);
    acc[stage] = fadd(acc[stage], fmul(in[i], in[i]));
  }
  const float inv =
      sfu_rsqrt(fadd(fmul(reduce8(acc), sfu_recip(static_cast<float>(size))),
                     1e-5f));
  for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] = fmul(weight[i], fmul(in[i], inv));
  }
}

static void fp8_quantize_group(ActTable& act, int group, const float* in) {
#pragma HLS INLINE
  float lane_max[kAccStages];
#pragma HLS ARRAY_PARTITION variable = lane_max complete dim = 1
  for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
    lane_max[stage] = 0.0f;
  }
  for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = lane_max inter false
    const float value = fabsf(in[lane]);
    const int stage = lane & (kAccStages - 1);
    if (value > lane_max[stage]) {
      lane_max[stage] = value;
    }
  }
  float wmax = lane_max[0];
  for (int stage = 1; stage < kAccStages; ++stage) {
    if (lane_max[stage] > wmax) {
      wmax = lane_max[stage];
    }
  }
  int8_t lanes[kQuantGroupSize];
#pragma HLS ARRAY_PARTITION variable = lanes complete dim = 1
  if (wmax == 0.0f) {
    for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS UNROLL
      lanes[lane] = 0;
    }
    pack_act_group(act, group, lanes, 1.0f);
  } else {
    const float scale = wmax / 127.0f;
    const float inv_scale = sfu_recip(scale);
    for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS PIPELINE II = 1
      int quantized = static_cast<int>(roundf(fmul(in[lane], inv_scale)));
      if (quantized > 127) quantized = 127;
      if (quantized < -127) quantized = -127;
      lanes[lane] = static_cast<int8_t>(quantized);
    }
    pack_act_group(act, group, lanes, scale);
  }
}

static void fp8_quantize(ActTable& act, const float* in, int size) {
#pragma HLS INLINE
  const int groups = size / kQuantGroupSize;
  for (int group = 0; group < groups; ++group) {
    fp8_quantize_group(act, group, in + group * kQuantGroupSize);
  }
}

static void emit_act(hls::stream<ActPkt>& acts, const ActTable& act, int groups) {
#pragma HLS INLINE
  for (int group = 0; group < groups; ++group) {
#pragma HLS PIPELINE II = 1
    ActPkt pkt;
    pkt.group = act.group[group];
    pkt.scale = act.scale[group];
    acts.write(pkt);
  }
}

static void load_act(hls::stream<ActPkt>& acts, ActTable& act, int groups) {
#pragma HLS INLINE
  for (int group = 0; group < groups; ++group) {
#pragma HLS PIPELINE II = 1
    const ActPkt pkt = acts.read();
    act.group[group] = pkt.group;
    act.scale[group] = pkt.scale;
  }
}

static float fp8_dot(const float* a, const float* b, int size) {
#pragma HLS INLINE
  float acc[kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
  for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
    acc[stage] = 0.0f;
  }
  for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
    const int stage = i & (kAccStages - 1);
    acc[stage] = fadd(acc[stage], fmul(a[i], b[i]));
  }
  return reduce8(acc);
}

static void fp8_conv(float* channels, ConvHist hist, int layer, int loop, int kind,
                     int channel_base,
                     const float weight[kConvKinds][kKeyDim][kConvSize],
                     int count, bool valid) {
#pragma HLS INLINE
  for (int c = 0; c < count; ++c) {
#pragma HLS PIPELINE II = 1
    const int ch = channel_base + c;
    float h0, h1, h2;
    unpack3(hist[layer][loop][kind][ch], h0, h1, h2);
    const float t0 = valid ? h0 : 0.0f;
    const float t1 = valid ? h1 : 0.0f;
    const float t2 = valid ? h2 : 0.0f;
    const float t3 = channels[c];
    const float sum =
        fadd(fadd(fmul(t0, weight[kind][ch][0]), fmul(t1, weight[kind][ch][1])),
             fadd(fmul(t2, weight[kind][ch][2]), fmul(t3, weight[kind][ch][3])));
    channels[c] = sum;
    hist[layer][loop][kind][ch] = pack3(t1, t2, t3);
  }
  for (int c = 0; c < count; ++c) {
    channels[c] = sfu_silu(channels[c]);
  }
}

static void fp8_l2scale(float* x, int size, float extra_scale) {
#pragma HLS INLINE
  float acc[kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
  for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
    acc[stage] = 0.0f;
  }
  for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
    const int stage = i & (kAccStages - 1);
    acc[stage] = fadd(acc[stage], fmul(x[i], x[i]));
  }
  const float inv = fmul(sfu_rsqrt(reduce8(acc) + 1e-6f), extra_scale);
  for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
    x[i] = fmul(x[i], inv);
  }
}

static void fp8_recurrence(StateMem S, int loop, int layer, int head, const float* q,
                           const float* k, const float* v, float beta,
                           float decay, bool valid, float* out) {
#pragma HLS INLINE off
  float prediction[kHeadVDim];
  float delta[kHeadVDim];
#pragma HLS ARRAY_PARTITION variable = prediction complete dim = 1
#pragma HLS ARRAY_PARTITION variable = delta complete dim = 1
#pragma HLS ARRAY_PARTITION variable = out complete dim = 1

  for (int g = 0; g < kHeadGroups; ++g) {
#pragma HLS LOOP_FLATTEN off
    float acc[kVecLanes][kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
    vec_clear(acc);
    for (int i = 0; i < kHeadKDim; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
      const int stage = i & (kAccStages - 1);
      const int addr = state_addr(loop, layer, head, i, g);
      float lane[kVecLanes];
#pragma HLS ARRAY_PARTITION variable = lane complete dim = 1
      s_load(S, addr, lane);
      for (int l = 0; l < kVecLanes; ++l) {
#pragma HLS UNROLL
        const float decayed = (valid ? lane[l] : 0.0f) * decay;
        lane[l] = decayed;
        acc[l][stage] = fadd(acc[l][stage], decayed * k[i]);
      }
      s_store(S, addr, lane);
    }
    for (int l = 0; l < kVecLanes; ++l) {
#pragma HLS UNROLL
      prediction[g * kVecLanes + l] = reduce8(acc[l]);
    }
  }

  for (int j = 0; j < kHeadVDim; ++j) {
#pragma HLS PIPELINE II = 1
    delta[j] = (v[j] - prediction[j]) * beta;
  }

  for (int g = 0; g < kHeadGroups; ++g) {
#pragma HLS LOOP_FLATTEN off
    float acc[kVecLanes][kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
    vec_clear(acc);
    for (int i = 0; i < kHeadKDim; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
      const int stage = i & (kAccStages - 1);
      const int addr = state_addr(loop, layer, head, i, g);
      float lane[kVecLanes];
#pragma HLS ARRAY_PARTITION variable = lane complete dim = 1
      s_load(S, addr, lane);
      for (int l = 0; l < kVecLanes; ++l) {
#pragma HLS UNROLL
        const float updated = fadd(lane[l], k[i] * delta[g * kVecLanes + l]);
        lane[l] = updated;
        acc[l][stage] = fadd(acc[l][stage], q[i] * updated);
      }
      s_store(S, addr, lane);
    }
    for (int l = 0; l < kVecLanes; ++l) {
#pragma HLS UNROLL
      out[g * kVecLanes + l] = reduce8(acc[l]);
    }
  }
}

static void fp8_head_norm_gate(float* x, const float* gate, const float* weight) {
#pragma HLS INLINE
  float acc[kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
  for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
    acc[stage] = 0.0f;
  }
  for (int i = 0; i < kHeadVDim; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
    const int stage = i & (kAccStages - 1);
    acc[stage] = fadd(acc[stage], fmul(x[i], x[i]));
  }
  const float inv =
      sfu_rsqrt(fadd(fmul(reduce8(acc), sfu_recip(static_cast<float>(kHeadVDim))),
                     1e-5f));
  for (int i = 0; i < kHeadVDim; ++i) {
    x[i] = fmul(weight[i], fmul(fmul(x[i], inv), sfu_silu(gate[i])));
  }
}

static void fp8_residual(float* x, const float* y, int size) {
#pragma HLS INLINE
  for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
    x[i] = fadd(x[i], y[i]);
  }
}

static void fp8_beta_decay(const float* norm, const float* b_proj,
                           const float* a_proj, float a_decay, float dt_bias,
                           float& beta, float& decay) {
#pragma HLS INLINE
  beta = sfu_sigmoid(fp8_dot(norm, b_proj, kDim));
  decay = sfu_exp(fmul(
      a_decay, sfu_softplus(fadd(fp8_dot(norm, a_proj, kDim), dt_bias))));
}

static void write_end(hls::stream<MemCmd>& mem_cmds,
                      hls::stream<ScratchCmd>& scratch_cmds,
                      hls::stream<RouteCmd>& route_cmds,
                      hls::stream<Q8Cmd>& q8_cmds, hls::stream<StageCmd>& beta_cmds,
                      hls::stream<StageCmd>& conv_cmds,
                      hls::stream<StageCmd>& rec_cmds, hls::stream<PostCmd>& post_cmds) {
#pragma HLS INLINE
  const CmdHeader end = make_hdr(kPhaseEnd, 0, 0, kWeightNone);
  MemCmd mem;
  mem.hdr = end;
  mem.kind = 0;
  mem.token = 0;
  mem_cmds.write(mem);
  ScratchCmd sc;
  sc.hdr = end;
  sc.kind = 0;
  scratch_cmds.write(sc);
  RouteCmd rt;
  rt.hdr = end;
  route_cmds.write(rt);
  Q8Cmd q8;
  q8.hdr = end;
  q8.kind = 0;
  q8_cmds.write(q8);
  StageCmd st;
  st.hdr = end;
  beta_cmds.write(st);
  conv_cmds.write(st);
  rec_cmds.write(st);
  PostCmd post;
  post.hdr = end;
  post.kind = 0;
  post_cmds.write(post);
}

static void schedule_process(int token, int reset_state, int loop_count,
                             hls::stream<MemCmd>& mem_cmds,
                             hls::stream<ScratchCmd>& scratch_cmds,
                             hls::stream<RouteCmd>& route_cmds,
                             hls::stream<Q8Cmd>& q8_cmds,
                             hls::stream<StageCmd>& beta_cmds,
                             hls::stream<StageCmd>& conv_cmds,
                             hls::stream<StageCmd>& rec_cmds,
                             hls::stream<PostCmd>& post_cmds,
                             hls::stream<int>& conv_reset,
                             hls::stream<int>& rec_reset) {
#pragma HLS INLINE off
  const int loops = clamp_loop_count(loop_count);
  conv_reset.write(reset_state);
  rec_reset.write(reset_state);

  MemCmd mem;
  mem.hdr = make_hdr(kPhaseBegin, 0, 0, kWeightNone);
  mem.kind = kMemEmbed;
  mem.token = token;
  mem_cmds.write(mem);
  PostCmd post;
  post.hdr = mem.hdr;
  post.kind = kPostEmbed;
  post_cmds.write(post);

  for (int layer = 0; layer < kNumLayers; ++layer) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 8
    const unsigned char layer_id = static_cast<unsigned char>(layer);

    for (int loop_i = 0; loop_i < loops; ++loop_i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4
      const unsigned char loop = static_cast<unsigned char>(loop_i);

      if (loop_i == 0) {
        mem.hdr = make_hdr(kPhaseLayer, layer_id, loop, kWeightNone);
        mem.kind = kMemSide;
        mem.token = 0;
        mem_cmds.write(mem);
      }

      ScratchCmd sc;
      sc.hdr = make_hdr(kPhaseLayer, layer_id, loop, kWeightMixerQkvg);
      if (loop_i == 0) {
        mem.hdr = sc.hdr;
        mem.kind = kMemFill;
        mem_cmds.write(mem);
        sc.kind = kScratchFill;
        scratch_cmds.write(sc);
      } else {
        sc.kind = kScratchRead;
        scratch_cmds.write(sc);
      }

      StageCmd st;
      st.hdr = make_hdr(kPhaseLayer, layer_id, loop, kWeightNone);
      beta_cmds.write(st);
      conv_cmds.write(st);
      rec_cmds.write(st);
      post.hdr = st.hdr;
      post.kind = kPostAttn;
      post_cmds.write(post);

      RouteCmd rt;
      rt.hdr = sc.hdr;
      route_cmds.write(rt);
      Q8Cmd q8;
      q8.hdr = rt.hdr;
      q8.kind = kQ8Qkvg;
      q8_cmds.write(q8);

      post.kind = kPostPackO;
      post_cmds.write(post);
      sc.hdr = make_hdr(kPhaseLayer, layer_id, loop, kWeightMixerO);
      sc.kind = kScratchORead;
      scratch_cmds.write(sc);
      rt.hdr = sc.hdr;
      route_cmds.write(rt);
      q8.hdr = rt.hdr;
      q8.kind = kQ8O;
      q8_cmds.write(q8);
      post.kind = kPostResO;
      post_cmds.write(post);

      if (loop_i == 0) {
        sc.hdr = make_hdr(kPhaseLayer, layer_id, loop, kWeightW13);
        sc.kind = kScratchRingBegin;
        scratch_cmds.write(sc);

        mem.hdr = sc.hdr;
        mem.kind = kMemFfnPrefetch;
        mem_cmds.write(mem);
      }
    }

    RouteCmd rt;
    Q8Cmd q8;
    post.hdr = make_hdr(kPhaseLayer, layer_id, 0, kWeightW13);
    post.kind = kPostFfn;
    post_cmds.write(post);

    ScratchCmd sc;
    sc.hdr = post.hdr;
    sc.kind = kScratchRingRead;
    scratch_cmds.write(sc);
    rt.hdr = sc.hdr;
    route_cmds.write(rt);
    q8.hdr = sc.hdr;
    q8.kind = kQ8W13;
    q8_cmds.write(q8);
    post.kind = kPostW13;
    post_cmds.write(post);

    sc.hdr = make_hdr(kPhaseLayer, layer_id, 0, kWeightW2);
    sc.kind = kScratchRingRead;
    scratch_cmds.write(sc);
    rt.hdr = sc.hdr;
    route_cmds.write(rt);
    q8.hdr = sc.hdr;
    q8.kind = kQ8W2;
    q8_cmds.write(q8);
    post.hdr = sc.hdr;
    post.kind = kPostResW2;
    post_cmds.write(post);
  }

  mem.hdr = make_hdr(kPhaseLm, 0, 0, kWeightLm);
  mem.kind = kMemLm;
  mem_cmds.write(mem);
  RouteCmd rt;
  rt.hdr = mem.hdr;
  route_cmds.write(rt);
  Q8Cmd q8;
  q8.hdr = mem.hdr;
  q8.kind = kQ8Lm;
  q8_cmds.write(q8);
  post.hdr = make_hdr(kPhaseLm, 0, 0, kWeightNone);
  post.kind = kPostFinal;
  post_cmds.write(post);
  mem.hdr = make_hdr(kPhaseLm, 0, 0, kWeightNone);
  mem.kind = kMemStore;
  mem_cmds.write(mem);

  write_end(mem_cmds, scratch_cmds, route_cmds, q8_cmds, beta_cmds, conv_cmds,
            rec_cmds, post_cmds);
}

static void axi_words(hls::stream<ParameterWord>& dst, const ParameterBeat* params,
                      int offset, int count) {
#pragma HLS INLINE
  for (int it = 0; it < count; ++it) {
#pragma HLS PIPELINE II = 4
    dst.write(read_word(params, offset + it));
  }
}

static void axi_broadcast(hls::stream<ParameterWord>& to_q8,
                          hls::stream<ParameterWord>& to_scratch,
                          const ParameterBeat* params, int offset, int count) {
#pragma HLS INLINE
  for (int it = 0; it < count; ++it) {
#pragma HLS PIPELINE II = 4
    const ParameterWord word = read_word(params, offset + it);
    to_q8.write(word);
    to_scratch.write(word);
  }
}

static void load_embedding(hls::stream<float>& out, const ParameterBeat* params,
                           int token) {
#pragma HLS INLINE
  const int row = token % kPackedRows;
  const int pair = row / 2;
  const int half = row & 1;
  const int base =
      kPackedTokOffset + token / kPackedRows * kDimGroups * kWordsPerGroup;
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
      out.write(static_cast<float>(value.to_int()) * scale);
    }
  }
}

static void memory_process(const ParameterBeat* params, const float* side,
                           uint32_t* next_token,
                           hls::stream<MemCmd>& cmds, hls::stream<float>& embed,
                           hls::stream<float>& rms_final, hls::stream<float>& rms_att,
                           hls::stream<float>& rms_ffn, hls::stream<float>& beta_side,
                           hls::stream<float>& conv_w, hls::stream<float>& o_norm,
                           hls::stream<ParameterWord>& fill_words,
                           hls::stream<ParameterWord>& axi_out,
                           hls::stream<uint32_t>& lm_token) {
#pragma HLS INLINE off
  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 28 max = 28
    const MemCmd cmd = cmds.read();
    if (cmd.hdr.phase == kPhaseEnd) {
      break;
    }
    const int layer = cmd.hdr.layer_id;
    if (cmd.kind == kMemEmbed) {
      load_embedding(embed, params, cmd.token);
      stream_write_vec(rms_final, side + kSideRmsFinalOffset, kDim);
    } else if (cmd.kind == kMemSide) {
      stream_write_vec(rms_att, side + SideRmsAttOffset(layer), kDim);
      stream_write_vec(rms_ffn, side + SideRmsFfnOffset(layer), kDim);
      stream_write_vec(beta_side, side + SideAProjOffset(layer),
                       kAProjSize + kBProjSize);
      stream_write_vec(beta_side, side + SideAOffset(layer), 2 * kNumHeads);
      stream_write_vec(conv_w, side + SideQConvOffset(layer),
                       kQConvSize + kKConvSize + kVConvSize);
      stream_write_vec(o_norm, side + SideONormOffset(layer), kHeadVDim);
    } else if (cmd.kind == kMemFill) {
      // QKVG is consumed while it fills. O is cached in its packed layout and
      // replayed head-wise after recurrent outputs become ready.
      axi_broadcast(axi_out, fill_words, params, PackedQOffset(layer, 0),
                    kPackedQKVGWords);
      axi_words(fill_words, params, PackedOOffset(layer), kPackedProjWords);
    } else if (cmd.kind == kMemFfnPrefetch) {
      axi_words(fill_words, params, PackedW13Offset(layer), kFfnPackedWords);
    } else if (cmd.kind == kMemLm) {
      axi_words(axi_out, params, kPackedTokOffset, kPackedTokWords);
    } else {
      *next_token = lm_token.read();
    }
  }
}

static ap_uint<14> ring_next(ap_uint<14> ptr) {
#pragma HLS INLINE
  const ap_uint<14> next = ptr + 1;
  if (next >= kFfnRingBase + kFfnRingWords) {
    return static_cast<ap_uint<14>>(kFfnRingBase);
  }
  return next;
}

static void ring_fill_step(
    ScratchMem scratch, hls::stream<ParameterWord>& fill_words,
    ap_uint<14>& wr_ptr, ap_uint<14>& occupancy, ap_uint<14>& produced) {
#pragma HLS INLINE
  if (produced == kFfnPackedWords || occupancy == kFfnRingWords) {
    return;
  }

  ParameterWord word;
  if (fill_words.read_nb(word)) {
    scratch_write(scratch, wr_ptr, word);
    wr_ptr = ring_next(wr_ptr);
    occupancy++;
    produced++;
  }
}

static void scratch_process(hls::stream<ScratchCmd>& cmds,
                            hls::stream<ParameterWord>& fill_words,
                            hls::stream<ParameterWord>& sram_words,
                            hls::stream<unsigned char>& ffn_ready) {
#pragma HLS INLINE off
  static ScratchMem scratch;
#pragma HLS BIND_STORAGE variable = scratch type = ram_s2p impl = uram
#pragma HLS ARRAY_PARTITION variable = scratch complete dim = 1

  ap_uint<14> wr_ptr = kFfnRingBase;
  ap_uint<14> rd_ptr = kFfnRingBase;
  ap_uint<14> occupancy = 0;
  ap_uint<14> produced = 0;
  bool ring_active = false;
  bool consumer_ready = false;

  for (;;) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 41 max = 89
    const ScratchCmd cmd = cmds.read();
    if (cmd.hdr.phase == kPhaseEnd) {
      break;
    }

    if (cmd.kind == kScratchFill) {
      for (int word = 0; word < kMixerCacheWords; ++word) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = scratch inter false
        scratch_write(scratch, static_cast<ap_uint<14>>(word), fill_words.read());
      }
    } else if (cmd.kind == kScratchRingBegin) {
      wr_ptr = kFfnRingBase;
      rd_ptr = kFfnRingBase;
      occupancy = 0;
      produced = 0;
      consumer_ready = false;
      ring_active = true;
    } else if (cmd.kind == kScratchRead) {
      ap_uint<14> read_addr = kMixerCacheBase;
      for (int sent = 0; sent < kPackedQKVGWords;) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = scratch inter false
#pragma HLS LOOP_TRIPCOUNT min = 4608 max = 12000
        if (sram_words.write_nb(scratch_read(scratch, read_addr))) {
          read_addr++;
          sent++;
        }
        if (ring_active) {
          ring_fill_step(scratch, fill_words, wr_ptr, occupancy, produced);
        }
      }
    } else if (cmd.kind == kScratchORead) {
      ap_uint<14> read_addr = kPackedQKVGWords;
      ap_uint<14> head_base = kPackedQKVGWords;
      ap_uint<4> head = 0;
      ap_uint<5> block = 0;
      ap_uint<4> word = 0;
      for (int sent = 0; sent < kPackedProjWords;) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = scratch inter false
#pragma HLS LOOP_TRIPCOUNT min = 1152 max = 5000
        if (sram_words.write_nb(scratch_read(scratch, read_addr))) {
          sent++;
          if (word == kWordsPerGroup - 1) {
            word = 0;
            if (block == kDim / kPackedRows - 1) {
              block = 0;
              head++;
              head_base += kWordsPerGroup;
              read_addr = head_base;
            } else {
              block++;
              read_addr += kValueGroups * kWordsPerGroup -
                           (kWordsPerGroup - 1);
            }
          } else {
            word++;
            read_addr++;
          }
        }
        if (ring_active) {
          ring_fill_step(scratch, fill_words, wr_ptr, occupancy, produced);
        }
      }
    } else {
      if (cmd.hdr.weight_kind == kWeightW13) {
        while (!consumer_ready) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = scratch inter false
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 12000
          unsigned char ready_layer;
          if (ffn_ready.read_nb(ready_layer)) {
            consumer_ready = ready_layer == cmd.hdr.layer_id;
          }
          ring_fill_step(scratch, fill_words, wr_ptr, occupancy, produced);
        }
      }

      const int word_count = cmd.hdr.weight_kind == kWeightW13
                                 ? 2 * kPackedW13Words
                                 : kPackedW2Words;
      for (int sent = 0; sent < word_count;) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = scratch inter false
#pragma HLS LOOP_TRIPCOUNT min = 3456 max = 16000
        const ap_uint<14> occupancy_before = occupancy;
        const bool was_full = occupancy_before == kFfnRingWords;
        // Keep producer and consumer addresses separated at the two boundary
        // cases. The ring never depends on same-cycle URAM read/write behavior.
        const bool can_accept =
            occupancy_before == 0 || (occupancy_before > 1 && !was_full);
        ParameterWord incoming;
        bool incoming_valid = false;
        if (can_accept && produced < kFfnPackedWords) {
          incoming_valid = fill_words.read_nb(incoming);
        }

        if (occupancy_before > 0) {
          sram_words.write(scratch_read(scratch, rd_ptr));
          rd_ptr = ring_next(rd_ptr);
          occupancy--;
          sent++;

          if (incoming_valid) {
            scratch_write(scratch, wr_ptr, incoming);
            wr_ptr = ring_next(wr_ptr);
            occupancy++;
            produced++;
          }
        } else if (incoming_valid) {
          sram_words.write(incoming);
          produced++;
          sent++;
        }
      }

      if (cmd.hdr.weight_kind == kWeightW2) {
        ring_active = false;
      }
    }
  }
}

static void weight_router(hls::stream<RouteCmd>& cmds,
                          hls::stream<ParameterWord>& axi_words,
                          hls::stream<ParameterWord>& sram_words,
                          hls::stream<ParameterWord>& weights) {
#pragma HLS INLINE off
  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 34 max = 82
    const RouteCmd cmd = cmds.read();
    if (cmd.hdr.phase == kPhaseEnd) {
      break;
    }
    const int count = cmd.hdr.weight_kind == kWeightMixerQkvg
                          ? kPackedQKVGWords
                      : cmd.hdr.weight_kind == kWeightMixerO
                          ? kPackedProjWords
                      : cmd.hdr.weight_kind == kWeightW13
                          ? 2 * kPackedW13Words
                      : cmd.hdr.weight_kind == kWeightW2 ? kPackedW2Words
                                                        : kPackedTokWords;
    const bool from_sram = cmd.hdr.weight_kind != kWeightLm &&
                           (cmd.hdr.weight_kind != kWeightMixerQkvg ||
                            cmd.hdr.loop_id != 0);
    if (from_sram) {
      for (int it = 0; it < count; ++it) {
#pragma HLS PIPELINE II = 1
        weights.write(sram_words.read());
      }
    } else {
      for (int it = 0; it < count; ++it) {
#pragma HLS PIPELINE II = 1
        weights.write(axi_words.read());
      }
    }
  }
}

static void q8_process(hls::stream<Q8Cmd>& cmds, hls::stream<ParameterWord>& weights,
                       hls::stream<ActPkt>& acts, hls::stream<FloatPair>& qkvg,
                       hls::stream<FloatPair>& o_vec,
                       hls::stream<FloatPair>& w13_vec,
                       hls::stream<FloatPair>& w2_vec,
                       hls::stream<uint32_t>& lm_token) {
#pragma HLS INLINE off
  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 34 max = 82
    const Q8Cmd cmd = cmds.read();
    if (cmd.hdr.phase == kPhaseEnd) {
      break;
    }
    const bool paired = cmd.kind == kQ8W13;
    const bool argmax_mode = cmd.kind == kQ8Lm;
    const bool head_o = cmd.kind == kQ8O;
    const int repeats = cmd.kind == kQ8Qkvg ? kNumHeads : 1;
    const int rows = cmd.kind == kQ8Qkvg ? kQkvgRows
                     : cmd.kind == kQ8W13 ? kHiddenDim
                     : cmd.kind == kQ8Lm  ? kVocabSize
                                          : kDim;
    const int groups = cmd.kind == kQ8O     ? 1
                       : cmd.kind == kQ8W2  ? kHiddenGroups
                                           : kDimGroups;
    const int count =
        head_o ? kPackedOHeadWords : q8_word_count(rows, groups, paired);
    ActTable act;
    if (!head_o) {
      load_act(acts, act, groups);
    }
    for (int r = 0; r < repeats; ++r) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 8
      uint32_t token = 0;
      q8_consume(weights, count, groups, act, acts, paired, argmax_mode, head_o,
                 cmd.kind, qkvg, o_vec, w13_vec, w2_vec, &token);
      if (argmax_mode) {
        lm_token.write(token);
      }
    }
  }
}

static void beta_process(hls::stream<StageCmd>& cmds, hls::stream<float>& beta_side,
                         hls::stream<float>& attn_norm, hls::stream<float>& betas) {
#pragma HLS INLINE off
  float a_proj[kNumHeads][kDim];
  float b_proj[kNumHeads][kDim];
  float a_decay[kNumHeads];
  float dt_bias[kNumHeads];
  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 9 max = 33
    const StageCmd cmd = cmds.read();
    if (cmd.hdr.phase == kPhaseEnd) {
      break;
    }
    float norm[kDim];
    if (cmd.hdr.loop_id == 0) {
      for (int h = 0; h < kNumHeads; ++h) {
        stream_read_vec(beta_side, a_proj[h], kDim);
      }
      for (int h = 0; h < kNumHeads; ++h) {
        stream_read_vec(beta_side, b_proj[h], kDim);
      }
      for (int h = 0; h < kNumHeads; ++h) {
#pragma HLS PIPELINE II = 1
        a_decay[h] = beta_side.read();
      }
      for (int h = 0; h < kNumHeads; ++h) {
#pragma HLS PIPELINE II = 1
        dt_bias[h] = beta_side.read();
      }
    }
    stream_read_vec(attn_norm, norm, kDim);
    for (int h = 0; h < kNumHeads; ++h) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 8
      float beta = 0.0f;
      float decay = 0.0f;
      fp8_beta_decay(norm, b_proj[h], a_proj[h], a_decay[h], dt_bias[h], beta,
                     decay);
      betas.write(beta);
      betas.write(decay);
    }
  }
}

static void conv_process(hls::stream<StageCmd>& cmds, hls::stream<int>& reset,
                         hls::stream<float>& q_scale_s, hls::stream<float>& conv_w_s,
                         hls::stream<FloatPair>& qkvg,
                         hls::stream<FloatPair>& convd) {
#pragma HLS INLINE off
  static ConvHist hist;
  static bool valid[kNumLayers][kMaxLoopCount][kNumHeads];
#pragma HLS BIND_STORAGE variable = hist type = ram_2p impl = bram
#pragma HLS ARRAY_PARTITION variable = valid complete dim = 0

  const int reset_state = reset.read();
  if (reset_state) {
    for (int layer = 0; layer < kNumLayers; ++layer) {
      for (int loop = 0; loop < kMaxLoopCount; ++loop) {
        for (int head = 0; head < kNumHeads; ++head) {
#pragma HLS UNROLL
          valid[layer][loop][head] = false;
        }
      }
    }
  }
  const float q_scale = q_scale_s.read();
  float conv_w[kConvKinds][kKeyDim][kConvSize];

  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 9 max = 33
    const StageCmd cmd = cmds.read();
    if (cmd.hdr.phase == kPhaseEnd) {
      break;
    }
    const int layer = cmd.hdr.layer_id;
    const int loop = cmd.hdr.loop_id;
    if (loop == 0) {
      for (int kind = 0; kind < kConvKinds; ++kind) {
        for (int c = 0; c < kKeyDim; ++c) {
          for (int t = 0; t < kConvSize; ++t) {
#pragma HLS PIPELINE II = 1
            conv_w[kind][c][t] = conv_w_s.read();
          }
        }
      }
    }
    for (int h = 0; h < kNumHeads; ++h) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 8
      float q[kHeadKDim];
      float k[kHeadKDim];
      float v[kHeadVDim];
      float g[kHeadVDim];
      stream_read_pairs(qkvg, q, kHeadKDim);
      stream_read_pairs(qkvg, k, kHeadKDim);
      stream_read_pairs(qkvg, v, kHeadVDim);
      stream_read_pairs(qkvg, g, kHeadVDim);
      const int base = h * kHeadKDim;
      const bool seen = valid[layer][loop][h];
      fp8_conv(q, hist, layer, loop, 0, base, conv_w, kHeadKDim, seen);
      fp8_conv(k, hist, layer, loop, 1, base, conv_w, kHeadKDim, seen);
      fp8_conv(v, hist, layer, loop, 2, base, conv_w, kHeadVDim, seen);
      fp8_l2scale(q, kHeadKDim, q_scale);
      fp8_l2scale(k, kHeadKDim, 1.0f);
      stream_write_pairs(convd, q, kHeadKDim);
      stream_write_pairs(convd, k, kHeadKDim);
      stream_write_pairs(convd, v, kHeadVDim);
      stream_write_pairs(convd, g, kHeadVDim);
      valid[layer][loop][h] = true;
    }
  }
}

static void rec_process(hls::stream<StageCmd>& cmds, hls::stream<int>& reset,
                        hls::stream<float>& o_norm_s,
                        hls::stream<FloatPair>& convd,
                        hls::stream<float>& betas,
                        hls::stream<FloatPair>& heads) {
#pragma HLS INLINE off
  static StateMem S;
  static bool valid[kNumLayers][kMaxLoopCount][kNumHeads];
#pragma HLS BIND_STORAGE variable = S type = ram_s2p impl = uram
#pragma HLS ARRAY_PARTITION variable = S complete dim = 1
#pragma HLS ARRAY_PARTITION variable = valid complete dim = 0

  const int reset_state = reset.read();
  float o_norm[kHeadVDim];
  if (reset_state) {
    for (int layer = 0; layer < kNumLayers; ++layer) {
      for (int loop = 0; loop < kMaxLoopCount; ++loop) {
        for (int head = 0; head < kNumHeads; ++head) {
#pragma HLS UNROLL
          valid[layer][loop][head] = false;
        }
      }
    }
  }

  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 9 max = 33
    const StageCmd cmd = cmds.read();
    if (cmd.hdr.phase == kPhaseEnd) {
      break;
    }
    const int layer = cmd.hdr.layer_id;
    const int loop = cmd.hdr.loop_id;
    if (loop == 0) {
      stream_read_vec(o_norm_s, o_norm, kHeadVDim);
    }
    for (int h = 0; h < kNumHeads; ++h) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 8
      float q[kHeadKDim];
      float k[kHeadKDim];
      float v[kHeadVDim];
      float g[kHeadVDim];
      float out[kHeadVDim];
      stream_read_pairs(convd, q, kHeadKDim);
      stream_read_pairs(convd, k, kHeadKDim);
      stream_read_pairs(convd, v, kHeadVDim);
      stream_read_pairs(convd, g, kHeadVDim);
      const float beta = betas.read();
      const float decay = betas.read();
      fp8_recurrence(S, loop, layer, h, q, k, v, beta, decay, valid[layer][loop][h],
                     out);
      fp8_head_norm_gate(out, g, o_norm);
      stream_write_pairs(heads, out, kHeadVDim);
      valid[layer][loop][h] = true;
    }
  }
}

static void post_process(hls::stream<PostCmd>& cmds, hls::stream<float>& embed,
                         hls::stream<float>& rms_final, hls::stream<float>& rms_att,
                         hls::stream<float>& rms_ffn,
                         hls::stream<FloatPair>& rec_heads,
                         hls::stream<FloatPair>& o_vec,
                         hls::stream<FloatPair>& w13_vec,
                         hls::stream<FloatPair>& w2_vec,
                         hls::stream<float>& q_scale_s,
                         hls::stream<float>& attn_norm, hls::stream<ActPkt>& acts,
                         hls::stream<unsigned char>& ffn_ready) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances = fmul limit = 16
#pragma HLS ALLOCATION operation instances = fadd limit = 16
  float x[kHiddenDim];
  float tmp[kHiddenDim];
  float fused[kHiddenDim];
  float weight[kDim];
  ActTable act;

  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 51 max = 123
    const PostCmd cmd = cmds.read();
    if (cmd.hdr.phase == kPhaseEnd) {
      break;
    }
    if (cmd.kind == kPostEmbed) {
      stream_read_vec(embed, x, kDim);
      q_scale_s.write(sfu_rsqrt(static_cast<float>(kHeadKDim)));
    } else if (cmd.kind == kPostAttn) {
      if (cmd.hdr.loop_id == 0) {
        stream_read_vec(rms_att, weight, kDim);
      }
      fp8_rmsnorm(tmp, x, weight, kDim);
      fp8_quantize(act, tmp, kDim);
      emit_act(acts, act, kDimGroups);
      stream_write_vec(attn_norm, tmp, kDim);
    } else if (cmd.kind == kPostPackO) {
      for (int h = 0; h < kNumHeads; ++h) {
        stream_read_pairs(rec_heads, tmp, kHeadVDim);
        fp8_quantize_group(act, 0, tmp);
        emit_act(acts, act, 1);
      }
    } else if (cmd.kind == kPostResO) {
      stream_read_pairs(o_vec, tmp, kDim);
      fp8_residual(x, tmp, kDim);
    } else if (cmd.kind == kPostFfn) {
      ffn_ready.write(cmd.hdr.layer_id);
      stream_read_vec(rms_ffn, weight, kDim);
      fp8_rmsnorm(tmp, x, weight, kDim);
      fp8_quantize(act, tmp, kDim);
      emit_act(acts, act, kDimGroups);
    } else if (cmd.kind == kPostW13) {
      stream_read_pairs(w13_vec, fused, kHiddenDim);
      fp8_quantize(act, fused, kHiddenDim);
      emit_act(acts, act, kHiddenGroups);
    } else if (cmd.kind == kPostResW2) {
      stream_read_pairs(w2_vec, tmp, kDim);
      fp8_residual(x, tmp, kDim);
    } else {
      stream_read_vec(rms_final, weight, kDim);
      fp8_rmsnorm(tmp, x, weight, kDim);
      fp8_quantize(act, tmp, kDim);
      emit_act(acts, act, kDimGroups);
    }
  }
}

} // namespace
} // namespace gdn

using namespace gdn;

extern "C" {

void decode(int token, int reset_state, int loop_count,
            const ParameterBeat* __restrict packed_params, const float* side,
            uint32_t* next_token) {
#pragma HLS INTERFACE m_axi port = packed_params bundle = params0 \
    depth = kPackedTotalBeats max_read_burst_length = 256 \
    num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = side bundle = gmem \
    depth = kSideFloatCount max_read_burst_length = 64 \
    num_read_outstanding = 4
#pragma HLS INTERFACE m_axi port = next_token bundle = gmem \
    depth = 1 num_write_outstanding = 2
#pragma HLS INTERFACE s_axilite port = token bundle = control
#pragma HLS INTERFACE s_axilite port = reset_state bundle = control
#pragma HLS INTERFACE s_axilite port = loop_count bundle = control
#pragma HLS INTERFACE s_axilite port = packed_params bundle = control
#pragma HLS INTERFACE s_axilite port = side bundle = control
#pragma HLS INTERFACE s_axilite port = next_token bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  hls::stream<MemCmd> mem_cmds("mem_cmds");
  hls::stream<ScratchCmd> scratch_cmds("scratch_cmds");
  hls::stream<RouteCmd> route_cmds("route_cmds");
  hls::stream<Q8Cmd> q8_cmds("q8_cmds");
  hls::stream<StageCmd> beta_cmds("beta_cmds");
  hls::stream<StageCmd> conv_cmds("conv_cmds");
  hls::stream<StageCmd> rec_cmds("rec_cmds");
  hls::stream<PostCmd> post_cmds("post_cmds");
  hls::stream<int> conv_reset("conv_reset");
  hls::stream<int> rec_reset("rec_reset");
  hls::stream<float> embed("embed");
  hls::stream<float> rms_final("rms_final");
  hls::stream<float> rms_att("rms_att");
  hls::stream<float> rms_ffn("rms_ffn");
  hls::stream<float> beta_side("beta_side");
  hls::stream<float> conv_w("conv_w");
  hls::stream<float> o_norm("o_norm");
  hls::stream<ParameterWord> fill_words("fill_words");
  hls::stream<ParameterWord> axi_words_s("axi_words");
  hls::stream<ParameterWord> sram_words("sram_words");
  hls::stream<ParameterWord> weights("weights");
  hls::stream<ActPkt> acts("acts");
  hls::stream<float> q_scale("q_scale");
  hls::stream<float> attn_norm("attn_norm");
  hls::stream<FloatPair> qkvg("qkvg");
  hls::stream<FloatPair> convd("convd");
  hls::stream<float> betas("betas");
  hls::stream<FloatPair> rec_heads("rec_heads");
  hls::stream<FloatPair> o_vec("o_vec");
  hls::stream<FloatPair> w13_vec("w13_vec");
  hls::stream<FloatPair> w2_vec("w2_vec");
  hls::stream<unsigned char> ffn_ready("ffn_ready");
  hls::stream<uint32_t> lm_token("lm_token");
#ifdef __SYNTHESIS__
#pragma HLS STREAM variable = mem_cmds depth = 16
#pragma HLS STREAM variable = scratch_cmds depth = 16
#pragma HLS STREAM variable = route_cmds depth = 16
#pragma HLS STREAM variable = q8_cmds depth = 16
#pragma HLS STREAM variable = beta_cmds depth = 16
#pragma HLS STREAM variable = conv_cmds depth = 16
#pragma HLS STREAM variable = rec_cmds depth = 16
#pragma HLS STREAM variable = post_cmds depth = 16
#pragma HLS BIND_STORAGE variable = mem_cmds type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = scratch_cmds type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = route_cmds type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = q8_cmds type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = beta_cmds type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = conv_cmds type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = rec_cmds type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = post_cmds type = fifo impl = srl
#else
#pragma HLS STREAM variable = mem_cmds depth = 256
#pragma HLS STREAM variable = scratch_cmds depth = 256
#pragma HLS STREAM variable = route_cmds depth = 256
#pragma HLS STREAM variable = q8_cmds depth = 256
#pragma HLS STREAM variable = beta_cmds depth = 256
#pragma HLS STREAM variable = conv_cmds depth = 256
#pragma HLS STREAM variable = rec_cmds depth = 256
#pragma HLS STREAM variable = post_cmds depth = 256
#endif
#pragma HLS STREAM variable = conv_reset depth = 2
#pragma HLS STREAM variable = rec_reset depth = 2
#pragma HLS STREAM variable = embed depth = 256
#pragma HLS STREAM variable = rms_final depth = 256
#pragma HLS STREAM variable = rms_att depth = 256
#pragma HLS STREAM variable = rms_ffn depth = 256
#pragma HLS STREAM variable = beta_side depth = 32
#pragma HLS STREAM variable = conv_w depth = 32
#pragma HLS STREAM variable = o_norm depth = 32
#pragma HLS STREAM variable = fill_words depth = 2
#pragma HLS STREAM variable = axi_words_s depth = 2
#pragma HLS STREAM variable = sram_words depth = 2
#pragma HLS STREAM variable = weights depth = 2
#pragma HLS BIND_STORAGE variable = fill_words type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = axi_words_s type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = sram_words type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = weights type = fifo impl = srl
#pragma HLS STREAM variable = acts depth = 24
#pragma HLS STREAM variable = q_scale depth = 2
#pragma HLS STREAM variable = attn_norm depth = 256
#pragma HLS STREAM variable = qkvg depth = 128
#pragma HLS STREAM variable = convd depth = 128
#pragma HLS STREAM variable = betas depth = 16
#pragma HLS STREAM variable = rec_heads depth = 32
#pragma HLS STREAM variable = o_vec depth = 128
#pragma HLS STREAM variable = w13_vec depth = 128
#pragma HLS STREAM variable = w2_vec depth = 128
#pragma HLS STREAM variable = ffn_ready depth = 2
#pragma HLS STREAM variable = lm_token depth = 2
#pragma HLS BIND_STORAGE variable = ffn_ready type = fifo impl = srl

#ifdef __SYNTHESIS__
  // One call site per frozen process. Synthesis builds parallel RTL;
  // C simulation uses the thread path below because g++ ignores DATAFLOW.
#pragma HLS DATAFLOW disable_start_propagation
  schedule_process(token, reset_state, loop_count, mem_cmds, scratch_cmds,
                   route_cmds, q8_cmds, beta_cmds, conv_cmds, rec_cmds,
                   post_cmds, conv_reset, rec_reset);
  memory_process(packed_params, side, next_token, mem_cmds, embed, rms_final,
                 rms_att, rms_ffn, beta_side, conv_w, o_norm, fill_words,
                 axi_words_s, lm_token);
  scratch_process(scratch_cmds, fill_words, sram_words, ffn_ready);
  weight_router(route_cmds, axi_words_s, sram_words, weights);
  q8_process(q8_cmds, weights, acts, qkvg, o_vec, w13_vec, w2_vec, lm_token);
  beta_process(beta_cmds, beta_side, attn_norm, betas);
  conv_process(conv_cmds, conv_reset, q_scale, conv_w, qkvg, convd);
  rec_process(rec_cmds, rec_reset, o_norm, convd, betas, rec_heads);
  post_process(post_cmds, embed, rms_final, rms_att, rms_ffn, rec_heads, o_vec,
               w13_vec, w2_vec, q_scale, attn_norm, acts, ffn_ready);
#else
  std::thread t_ctrl([&] {
    schedule_process(token, reset_state, loop_count, mem_cmds, scratch_cmds,
                     route_cmds, q8_cmds, beta_cmds, conv_cmds, rec_cmds,
                     post_cmds, conv_reset, rec_reset);
  });
  std::thread t_mem([&] {
    memory_process(packed_params, side, next_token, mem_cmds, embed, rms_final,
                   rms_att, rms_ffn, beta_side, conv_w, o_norm, fill_words,
                   axi_words_s, lm_token);
  });
  std::thread t_scratch([&] {
    scratch_process(scratch_cmds, fill_words, sram_words, ffn_ready);
  });
  std::thread t_router(
      [&] { weight_router(route_cmds, axi_words_s, sram_words, weights); });
  std::thread t_q8([&] {
    q8_process(q8_cmds, weights, acts, qkvg, o_vec, w13_vec, w2_vec, lm_token);
  });
  std::thread t_beta(
      [&] { beta_process(beta_cmds, beta_side, attn_norm, betas); });
  std::thread t_conv(
      [&] { conv_process(conv_cmds, conv_reset, q_scale, conv_w, qkvg, convd); });
  std::thread t_rec([&] {
    rec_process(rec_cmds, rec_reset, o_norm, convd, betas, rec_heads);
  });
  std::thread t_post([&] {
    post_process(post_cmds, embed, rms_final, rms_att, rms_ffn, rec_heads, o_vec,
                 w13_vec, w2_vec, q_scale, attn_norm, acts, ffn_ready);
  });
  t_ctrl.join();
  t_mem.join();
  t_scratch.join();
  t_router.join();
  t_q8.join();
  t_beta.join();
  t_conv.join();
  t_rec.join();
  t_post.join();
#endif
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

void CpuForward(RunState& s, const Weights& w, int token, float* logits,
                int loop_count) {
  const int loops = loop_count < 1 ? 1 : (loop_count > kMaxLoopCount ? kMaxLoopCount : loop_count);
  const std::size_t q_base = static_cast<std::size_t>(token) * kDim;
  const std::size_t s_base = static_cast<std::size_t>(token) * kDimGroups;
  for (int i = 0; i < kDim; ++i) {
    s.x[i] = static_cast<float>(w.tok_emb.q[q_base + i]) *
             w.tok_emb.s[s_base + i / kQuantGroupSize];
  }

  const float q_scale = 1.0f / std::sqrt(static_cast<float>(kHeadKDim));

  for (int layer = 0; layer < kNumLayers; ++layer) {
    const LayerWeights& l = w.layers[layer];

    for (int loop = 0; loop < loops; ++loop) {
      RmsNorm(s.xb.data(), s.x.data(), l.attn_norm.data(), kDim);
      Quantize(s.xq_q.data(), s.xq_s.data(), s.xb.data(), kDim);
      Matmul(s.q.data(), s.xq_q.data(), s.xq_s.data(), l.q_proj);
      Matmul(s.k.data(), s.xq_q.data(), s.xq_s.data(), l.k_proj);
      Matmul(s.v.data(), s.xq_q.data(), s.xq_s.data(), l.v_proj);
      Matmul(s.gate.data(), s.xq_q.data(), s.xq_s.data(), l.g_proj);

      const std::size_t slot =
          static_cast<std::size_t>(loop) * kNumLayers + static_cast<std::size_t>(layer);
      ConvStep(s.q.data(), s.q_conv_state.data() + slot * kQConvSize, l.q_conv.data(),
               kKeyDim);
      ConvStep(s.k.data(), s.k_conv_state.data() + slot * kKConvSize, l.k_conv.data(),
               kKeyDim);
      ConvStep(s.v.data(), s.v_conv_state.data() + slot * kVConvSize, l.v_conv.data(),
               kValueDim);

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

      float* S_layer = s.S.data() + slot * kNumHeads * kHeadKDim * kHeadVDim;
      for (int h = 0; h < kNumHeads; ++h) {
        float* S = S_layer + static_cast<std::size_t>(h) * kHeadKDim * kHeadVDim;
        const float* qh = s.q.data() + h * kHeadKDim;
        const float* kh = s.k.data() + h * kHeadKDim;
        const float* vh = s.v.data() + h * kHeadVDim;
        float* out = s.linear_out.data() + h * kHeadVDim;

        for (int i = 0; i < kHeadKDim * kHeadVDim; ++i) S[i] *= s.decay[h];
        for (int j = 0; j < kHeadVDim; ++j) {
          float prediction = 0.0f;
          for (int i = 0; i < kHeadKDim; ++i)
            prediction += S[i * kHeadVDim + j] * kh[i];
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
    }

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

int Decode(int token, bool reset_state, int loop_count, cl::CommandQueue& q,
           cl::Kernel& kernel, std::uint32_t* next_token,
           cl::Buffer& next_token_buffer) {
  cl_int err = CL_SUCCESS;
  err = kernel.setArg(0, token);
  if (err != CL_SUCCESS) return -1;
  err = kernel.setArg(1, reset_state ? 1 : 0);
  if (err != CL_SUCCESS) return -1;
  err = kernel.setArg(2, loop_count);
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
