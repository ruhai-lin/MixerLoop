#include "vocab.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace gdn {

Tokenizer LoadTokenizer(const std::string& path, int vocab_size) {
  std::ifstream fs(path, std::ios::binary);
  if (!fs) {
    throw std::runtime_error("could not open tokenizer: " + path);
  }

  Tokenizer tokenizer;
  tokenizer.vocab.resize(vocab_size);
  tokenizer.scores.resize(vocab_size);
  for (int i = 0; i < 256; ++i) {
    tokenizer.byte_pieces[i] = std::string(1, static_cast<char>(i));
  }

  fs.read(reinterpret_cast<char*>(&tokenizer.max_token_length), sizeof(int));
  if (!fs || tokenizer.max_token_length <= 0) {
    throw std::runtime_error("failed to read tokenizer max token length");
  }

  for (int i = 0; i < vocab_size; ++i) {
    int len = 0;
    fs.read(reinterpret_cast<char*>(&tokenizer.scores[i]), sizeof(float));
    fs.read(reinterpret_cast<char*>(&len), sizeof(int));
    if (!fs || len < 0 || len > tokenizer.max_token_length * 4) {
      throw std::runtime_error("bad tokenizer token metadata");
    }
    tokenizer.vocab[i].resize(len);
    fs.read(tokenizer.vocab[i].data(), len);
    if (!fs) {
      throw std::runtime_error("failed to read tokenizer token bytes");
    }
  }
  return tokenizer;
}

static void EnsureSorted(Tokenizer& tokenizer) {
  if (!tokenizer.sorted.empty()) {
    return;
  }
  tokenizer.sorted.reserve(tokenizer.vocab.size());
  for (int i = 0; i < static_cast<int>(tokenizer.vocab.size()); ++i) {
    tokenizer.sorted.push_back({tokenizer.vocab[i], i});
  }
  std::sort(tokenizer.sorted.begin(), tokenizer.sorted.end(),
            [](const TokenIndex& a, const TokenIndex& b) {
              return a.str < b.str;
            });
}

static int Lookup(const Tokenizer& tokenizer, const std::string& text) {
  auto it = std::lower_bound(
      tokenizer.sorted.begin(), tokenizer.sorted.end(), text,
      [](const TokenIndex& entry, const std::string& value) {
        return entry.str < value;
      });
  if (it != tokenizer.sorted.end() && it->str == text) {
    return it->id;
  }
  return -1;
}

std::vector<int> Encode(Tokenizer& tokenizer, const std::string& text, bool bos,
                        bool eos) {
  EnsureSorted(tokenizer);
  std::vector<int> tokens;
  tokens.reserve(text.size() + 3);
  if (bos) {
    tokens.push_back(1);
  }

  // Llama sentencepiece adds a dummy leading space so that the first word is
  // treated identically to interior words (matches runq.c encode()).
  if (!text.empty()) {
    const int dummy_prefix = Lookup(tokenizer, " ");
    if (dummy_prefix >= 0) {
      tokens.push_back(dummy_prefix);
    }
  }

  for (std::size_t i = 0; i < text.size();) {
    std::size_t len = 1;
    while (i + len < text.size() &&
           (static_cast<unsigned char>(text[i + len]) & 0xC0) == 0x80) {
      ++len;
    }
    const std::string piece = text.substr(i, len);
    const int id = Lookup(tokenizer, piece);
    if (id >= 0) {
      tokens.push_back(id);
    } else {
      // byte_fallback: emit each raw byte as token (byte value + 3), since the
      // first three vocab entries are <unk>, <s>, </s>.
      for (unsigned char byte : piece) {
        tokens.push_back(static_cast<int>(byte) + 3);
      }
    }
    i += len;
  }

  while (tokens.size() >= 2) {
    float best_score = -1e10f;
    int best_id = -1;
    std::size_t best_idx = 0;
    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
      const std::string merged =
          tokenizer.vocab[tokens[i]] + tokenizer.vocab[tokens[i + 1]];
      const int id = Lookup(tokenizer, merged);
      if (id >= 0 && tokenizer.scores[id] > best_score) {
        best_score = tokenizer.scores[id];
        best_id = id;
        best_idx = i;
      }
    }
    if (best_id < 0) {
      break;
    }
    tokens[best_idx] = best_id;
    tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(best_idx + 1));
  }

  if (eos) {
    tokens.push_back(2);
  }
  return tokens;
}

std::string DecodePiece(const Tokenizer& tokenizer, int prev_token, int token) {
  std::string piece = tokenizer.vocab.at(token);
  if (prev_token == 1 && !piece.empty() && piece.front() == ' ') {
    piece.erase(piece.begin());
  }

  unsigned int byte_val = 0;
  if (std::sscanf(piece.c_str(), "<0x%02X>", &byte_val) == 1 &&
      byte_val < 256) {
    return tokenizer.byte_pieces[byte_val];
  }
  return piece;
}

} // namespace gdn
