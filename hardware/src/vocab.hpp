#ifndef GDN_VOCAB_HPP_
#define GDN_VOCAB_HPP_

#include <string>
#include <vector>

namespace gdn {

struct TokenIndex {
  std::string str;
  int id;
};

struct Tokenizer {
  std::vector<std::string> vocab;
  std::vector<float> scores;
  std::vector<TokenIndex> sorted;
  std::string byte_pieces[256];
  int max_token_length = 0;
};

Tokenizer LoadTokenizer(const std::string& path, int vocab_size);
std::vector<int> Encode(Tokenizer& tokenizer, const std::string& text, bool bos,
                        bool eos);
std::string DecodePiece(const Tokenizer& tokenizer, int prev_token, int token);

} // namespace gdn

#endif // GDN_VOCAB_HPP_
