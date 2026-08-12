#include "weight.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace gdn {
namespace {

struct CheckpointHeader {
  int dim;
  int hidden_dim;
  int n_layers;
  int num_heads;
  int head_k_dim;
  int head_v_dim;
  int conv_size;
  int vocab_size;
  int seq_len;
  int shared_classifier;
  int pad;
};

void ReadFp32(std::ifstream& fs, std::vector<float>& out, std::size_t count,
              const char* name) {
  out.resize(count);
  fs.read(reinterpret_cast<char*>(out.data()),
          static_cast<std::streamsize>(count * sizeof(float)));
  if (!fs) {
    throw std::runtime_error(std::string("failed reading fp32 tensor: ") + name);
  }
}

QuantizedTensor ReadQ8(std::ifstream& fs, int rows, int cols, const char* name) {
  QuantizedTensor t;
  t.rows = rows;
  t.cols = cols;
  t.groups = cols / kQuantGroupSize;
  t.q.resize(static_cast<std::size_t>(rows) * cols);
  t.s.resize(static_cast<std::size_t>(rows) * t.groups);
  fs.read(reinterpret_cast<char*>(t.q.data()),
          static_cast<std::streamsize>(t.q.size()));
  fs.read(reinterpret_cast<char*>(t.s.data()),
          static_cast<std::streamsize>(t.s.size() * sizeof(float)));
  if (!fs) {
    throw std::runtime_error(std::string("failed reading q8 tensor: ") + name);
  }
  return t;
}

void AppendWord(std::vector<std::uint8_t>& out,
                const std::uint8_t word[kPackedWordBytes]) {
  out.insert(out.end(), word, word + kPackedWordBytes);
}

void PackGroup(const QuantizedTensor& m, int row0, int group,
               std::vector<std::uint8_t>& out) {
  const int groups = m.groups;
  std::uint8_t word[kPackedWordBytes]{};
  for (int row = 0; row < kPackedRows; ++row) {
    const float scale =
        m.s[static_cast<std::size_t>(row0 + row) * groups + group];
    std::memcpy(word + row * 4, &scale, sizeof(float));
  }
  AppendWord(out, word);

  for (int pair = 0; pair < kPackedRows / 2; ++pair) {
    std::memset(word, 0, sizeof(word));
    for (int half = 0; half < 2; ++half) {
      const int row = row0 + pair * 2 + half;
      const std::size_t q_base =
          static_cast<std::size_t>(row) * m.cols + group * kQuantGroupSize;
      for (int lane = 0; lane < kQuantGroupSize; ++lane) {
        word[half * kQuantGroupSize + lane] =
            static_cast<std::uint8_t>(m.q[q_base + lane]);
      }
    }
    AppendWord(out, word);
  }
}

void PackRows(const QuantizedTensor& m, int first_row, int rows,
              std::vector<std::uint8_t>& out) {
  for (int row = first_row; row < first_row + rows; row += kPackedRows) {
    for (int group = 0; group < m.groups; ++group) {
      PackGroup(m, row, group, out);
    }
  }
}

void PackMatrix(const QuantizedTensor& m, std::vector<std::uint8_t>& out) {
  PackRows(m, 0, m.rows, out);
}

void PackPair(const QuantizedTensor& a, const QuantizedTensor& b,
              std::vector<std::uint8_t>& out) {
  for (int row = 0; row < a.rows; row += kPackedRows) {
    for (int group = 0; group < a.groups; ++group) {
      PackGroup(a, row, group, out);
      PackGroup(b, row, group, out);
    }
  }
}

void AppendFloats(std::vector<float>& out, const std::vector<float>& src) {
  out.insert(out.end(), src.begin(), src.end());
}

} // namespace

void LoadWeights(Weights& w, const std::string& path) {
  std::ifstream fs(path, std::ios::binary);
  if (!fs) {
    throw std::runtime_error("could not open checkpoint: " + path);
  }

  std::uint32_t magic = 0;
  int version = 0;
  CheckpointHeader h{};
  int group_size = 0;
  fs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  fs.read(reinterpret_cast<char*>(&version), sizeof(version));
  fs.read(reinterpret_cast<char*>(&h), sizeof(h));
  fs.read(reinterpret_cast<char*>(&group_size), sizeof(group_size));
  if (!fs) {
    throw std::runtime_error("failed to read checkpoint header");
  }
  if (magic != kCheckpointMagic || version != kCheckpointVersion) {
    throw std::runtime_error("bad GDN Q8 checkpoint magic/version");
  }
  if (h.dim != kDim || h.hidden_dim != kHiddenDim || h.n_layers != kNumLayers ||
      h.num_heads != kNumHeads || h.head_k_dim != kHeadKDim ||
      h.head_v_dim != kHeadVDim || h.conv_size != kConvSize ||
      h.vocab_size != kVocabSize || h.shared_classifier != 1 ||
      group_size != kQuantGroupSize) {
    throw std::runtime_error("checkpoint config does not match gdn.hls constants");
  }
  if (h.seq_len != kSeqLen) {
    throw std::runtime_error("checkpoint seq_len does not match M0 contract");
  }

  fs.seekg(kCheckpointHeaderBytes, std::ios::beg);

  w.tok_emb = ReadQ8(fs, kVocabSize, kDim, "embedding");

  w.layers.resize(kNumLayers);
  for (int layer = 0; layer < kNumLayers; ++layer) {
    LayerWeights& l = w.layers[layer];
    ReadFp32(fs, l.attn_norm, kDim, "attn_norm");
    l.q_proj = ReadQ8(fs, kKeyDim, kDim, "q_proj");
    l.k_proj = ReadQ8(fs, kKeyDim, kDim, "k_proj");
    l.v_proj = ReadQ8(fs, kValueDim, kDim, "v_proj");
    ReadFp32(fs, l.a_proj, kAProjSize, "a_proj");
    ReadFp32(fs, l.b_proj, kBProjSize, "b_proj");
    l.g_proj = ReadQ8(fs, kValueDim, kDim, "g_proj");
    ReadFp32(fs, l.q_conv, kQConvSize, "q_conv");
    ReadFp32(fs, l.k_conv, kKConvSize, "k_conv");
    ReadFp32(fs, l.v_conv, kVConvSize, "v_conv");
    ReadFp32(fs, l.A, kNumHeads, "A");
    ReadFp32(fs, l.dt_bias, kNumHeads, "dt_bias");
    ReadFp32(fs, l.o_norm, kHeadVDim, "o_norm");
    l.o_proj = ReadQ8(fs, kDim, kValueDim, "o_proj");
    ReadFp32(fs, l.ffn_norm, kDim, "ffn_norm");
    l.w1 = ReadQ8(fs, kHiddenDim, kDim, "w1");
    l.w2 = ReadQ8(fs, kDim, kHiddenDim, "w2");
    l.w3 = ReadQ8(fs, kHiddenDim, kDim, "w3");
  }
  ReadFp32(fs, w.rms_final, kDim, "rms_final");
}

std::vector<std::uint8_t> PackParameters(const Weights& w) {
  std::vector<std::uint8_t> blob;
  blob.reserve(kPackedTotalBytes);
  PackMatrix(w.tok_emb, blob);
  for (int layer = 0; layer < kNumLayers; ++layer) {
    const LayerWeights& l = w.layers[layer];
    for (int head = 0; head < kNumHeads; ++head) {
      PackRows(l.q_proj, head * kHeadKDim, kHeadKDim, blob);
      PackRows(l.k_proj, head * kHeadKDim, kHeadKDim, blob);
      PackRows(l.v_proj, head * kHeadVDim, kHeadVDim, blob);
      PackRows(l.g_proj, head * kHeadVDim, kHeadVDim, blob);
    }
    PackMatrix(l.o_proj, blob);
    PackPair(l.w1, l.w3, blob);
    PackMatrix(l.w2, blob);
  }
  if (blob.size() != kPackedTotalBytes) {
    throw std::runtime_error("packed parameter size mismatch");
  }
  return blob;
}

std::vector<float> BuildFp32Side(const Weights& w) {
  std::vector<float> side;
  side.reserve(kSideFloatCount);
  for (int layer = 0; layer < kNumLayers; ++layer) {
    AppendFloats(side, w.layers[layer].attn_norm);
  }
  for (int layer = 0; layer < kNumLayers; ++layer) {
    AppendFloats(side, w.layers[layer].ffn_norm);
  }
  AppendFloats(side, w.rms_final);
  for (int layer = 0; layer < kNumLayers; ++layer) {
    const LayerWeights& l = w.layers[layer];
    AppendFloats(side, l.a_proj);
    AppendFloats(side, l.b_proj);
    AppendFloats(side, l.q_conv);
    AppendFloats(side, l.k_conv);
    AppendFloats(side, l.v_conv);
    AppendFloats(side, l.A);
    AppendFloats(side, l.dt_bias);
    AppendFloats(side, l.o_norm);
  }
  if (side.size() != static_cast<std::size_t>(kSideFloatCount)) {
    throw std::runtime_error("fp32 side size mismatch");
  }
  return side;
}

} // namespace gdn
