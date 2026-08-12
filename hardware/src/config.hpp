#ifndef GDN_CONFIG_HPP_
#define GDN_CONFIG_HPP_

#include <cstddef>
#include <cstdint>

namespace gdn {

// Model.
constexpr int kDim = 256;
constexpr int kHiddenDim = 768;
constexpr int kNumLayers = 8;
constexpr int kNumHeads = 8;
constexpr int kHeadKDim = 32;
constexpr int kHeadVDim = 32;
constexpr int kConvSize = 4;
constexpr int kVocabSize = 32000;
constexpr int kSeqLen = 1024;
constexpr int kKeyDim = kNumHeads * kHeadKDim;
constexpr int kValueDim = kNumHeads * kHeadVDim;

// Checkpoint and Q8 packing.
constexpr std::uint32_t kCheckpointMagic = 0x47444e65u; // "GDNe"
constexpr int kCheckpointVersion = 2;
constexpr int kCheckpointHeaderBytes = 256;
constexpr int kQuantGroupSize = 32;
constexpr int kDimGroups = kDim / kQuantGroupSize;
constexpr int kValueGroups = kValueDim / kQuantGroupSize;
constexpr int kHiddenGroups = kHiddenDim / kQuantGroupSize;
constexpr int kPackedRows = 16;
constexpr int kWordsPerGroup = 1 + kPackedRows / 2;
constexpr int kPackedWordBytes = 64;
constexpr int kBeatsPerWord = 4;

constexpr int PackedMatrixWords(int rows, int groups) {
  return rows / kPackedRows * groups * kWordsPerGroup;
}

constexpr int kPackedTokWords = PackedMatrixWords(kVocabSize, kDimGroups);
constexpr int kPackedProjWords = PackedMatrixWords(kDim, kDimGroups);
constexpr int kPackedHeadWords = PackedMatrixWords(kHeadKDim, kDimGroups);
constexpr int kPackedQKVGWords = 4 * kNumHeads * kPackedHeadWords;
constexpr int kPackedW13Words = PackedMatrixWords(kHiddenDim, kDimGroups);
constexpr int kPackedW2Words = PackedMatrixWords(kDim, kHiddenGroups);

constexpr int PackedLayerBase(int layer) {
  return kPackedTokWords +
         layer * (kPackedQKVGWords + kPackedProjWords +
                  2 * kPackedW13Words + kPackedW2Words);
}
constexpr int PackedQOffset(int layer, int head) {
  return PackedLayerBase(layer) + 4 * head * kPackedHeadWords;
}
constexpr int PackedKOffset(int layer, int head) {
  return PackedQOffset(layer, head) + kPackedHeadWords;
}
constexpr int PackedVOffset(int layer, int head) {
  return PackedKOffset(layer, head) + kPackedHeadWords;
}
constexpr int PackedGOffset(int layer, int head) {
  return PackedVOffset(layer, head) + kPackedHeadWords;
}
constexpr int PackedOOffset(int layer) {
  return PackedLayerBase(layer) + kPackedQKVGWords;
}
constexpr int PackedW13Offset(int layer) {
  return PackedOOffset(layer) + kPackedProjWords;
}
constexpr int PackedW2Offset(int layer) {
  return PackedW13Offset(layer) + 2 * kPackedW13Words;
}

constexpr int kPackedTokOffset = 0;
constexpr std::size_t kPackedTotalBytes =
    static_cast<std::size_t>(PackedLayerBase(kNumLayers)) * kPackedWordBytes;

// FP32 side parameters.
constexpr int kAProjSize = kNumHeads * kDim;
constexpr int kBProjSize = kNumHeads * kDim;
constexpr int kQConvSize = kKeyDim * kConvSize;
constexpr int kKConvSize = kKeyDim * kConvSize;
constexpr int kVConvSize = kValueDim * kConvSize;

constexpr int SideRmsAttOffset(int layer) { return layer * kDim; }
constexpr int SideRmsFfnOffset(int layer) {
  return (kNumLayers + layer) * kDim;
}
constexpr int kSideRmsFinalOffset = 2 * kNumLayers * kDim;
constexpr int SideLayerBase(int layer) {
  return kSideRmsFinalOffset + kDim +
         layer * (kAProjSize + kBProjSize + kQConvSize + kKConvSize +
                  kVConvSize + 2 * kNumHeads + kHeadVDim);
}
constexpr int SideAProjOffset(int layer) { return SideLayerBase(layer); }
constexpr int SideBProjOffset(int layer) {
  return SideAProjOffset(layer) + kAProjSize;
}
constexpr int SideQConvOffset(int layer) {
  return SideBProjOffset(layer) + kBProjSize;
}
constexpr int SideKConvOffset(int layer) {
  return SideQConvOffset(layer) + kQConvSize;
}
constexpr int SideVConvOffset(int layer) {
  return SideKConvOffset(layer) + kKConvSize;
}
constexpr int SideAOffset(int layer) {
  return SideVConvOffset(layer) + kVConvSize;
}
constexpr int SideDtBiasOffset(int layer) {
  return SideAOffset(layer) + kNumHeads;
}
constexpr int SideONormOffset(int layer) {
  return SideDtBiasOffset(layer) + kNumHeads;
}
constexpr int kSideFloatCount = SideLayerBase(kNumLayers);

constexpr std::size_t kSStateCount = static_cast<std::size_t>(kNumLayers) *
                                     kNumHeads * kHeadKDim * kHeadVDim;
constexpr std::size_t kQConvStateCount =
    static_cast<std::size_t>(kNumLayers) * kKeyDim * kConvSize;
constexpr std::size_t kKConvStateCount = kQConvStateCount;
constexpr std::size_t kVConvStateCount =
    static_cast<std::size_t>(kNumLayers) * kValueDim * kConvSize;
constexpr int kStateBanks = 16;
constexpr int kStateGroups = kHeadVDim / kStateBanks;

} // namespace gdn

#endif // GDN_CONFIG_HPP_
