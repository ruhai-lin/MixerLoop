#ifndef GDN_WEIGHT_HPP_
#define GDN_WEIGHT_HPP_

#include "config.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gdn {

// Split Q8_0 tensor: symmetric int8 values plus one fp32 scale per 32-lane
// group, grouped along each matrix row (the layout produced by export.py).
struct QuantizedTensor {
  std::vector<std::int8_t> q; // [rows * cols]
  std::vector<float> s;       // [rows * (cols / group_size)]
  int rows = 0;
  int cols = 0;
  int groups = 0;
};

struct LayerWeights {
  std::vector<float> attn_norm; // [dim]
  QuantizedTensor q_proj;       // [key_dim, dim]
  QuantizedTensor k_proj;       // [key_dim, dim]
  QuantizedTensor v_proj;       // [value_dim, dim]
  std::vector<float> a_proj;    // [heads, dim]
  std::vector<float> b_proj;    // [heads, dim]
  QuantizedTensor g_proj;       // [value_dim, dim]
  std::vector<float> q_conv;    // [key_dim, conv_size]
  std::vector<float> k_conv;    // [key_dim, conv_size]
  std::vector<float> v_conv;    // [value_dim, conv_size]
  std::vector<float> A;         // [heads]
  std::vector<float> dt_bias;   // [heads]
  std::vector<float> o_norm;    // [head_v_dim]
  QuantizedTensor o_proj;       // [dim, value_dim]
  std::vector<float> ffn_norm;  // [dim]
  QuantizedTensor w1;           // [hidden, dim]
  QuantizedTensor w2;           // [dim, hidden]
  QuantizedTensor w3;           // [hidden, dim]
};

struct Weights {
  QuantizedTensor tok_emb;          // [vocab, dim] (shared classifier / wcls)
  std::vector<LayerWeights> layers; // [n_layers]
  std::vector<float> rms_final;     // [dim]
};

void LoadWeights(Weights& weights, const std::string& path);

// HLS host helpers: build the canonical single-HP 512-bit parameter blob and
// the fp32 side array consumed by decode.
std::vector<std::uint8_t> PackParameters(const Weights& weights);
std::vector<float> BuildFp32Side(const Weights& weights);

} // namespace gdn

#endif // GDN_WEIGHT_HPP_
