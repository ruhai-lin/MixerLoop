// Functional cross-check of the HLS `decode` kernel against the CPU reference.
//
// Builds the exact buffers the FPGA host would build (single-HP parameters,
// fp32 side array), runs the synthesizable kernel
// natively, and compares its argmax token against CpuForward for each step.
// This catches datapath bugs in seconds instead of one bitstream per attempt.

#include "decode.hpp"
#include "weight.hpp"

#include <ap_int.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" void decode(int token, int reset_state,
                       const ap_uint<128>* packed_params, const float* side,
                       std::uint32_t* next_token);

namespace {

std::vector<ap_uint<128>> ToBeats(const std::vector<std::uint8_t>& bytes) {
  std::vector<ap_uint<128>> beats(bytes.size() / 16);
  for (std::size_t i = 0; i < beats.size(); ++i) {
    ap_uint<128> value = 0;
    for (int b = 0; b < 16; ++b) {
      value.range(b * 8 + 7, b * 8) = bytes[i * 16 + b];
    }
    beats[i] = value;
  }
  return beats;
}

} // namespace

int main(int argc, char** argv) {
  const std::string weight_path =
      argc > 1 ? argv[1] : "model/climbmix15M_demo_q8.bin";
  const int steps = argc > 2 ? std::atoi(argv[2]) : 8;

  gdn::Weights weights;
  gdn::LoadWeights(weights, weight_path);
  std::printf("loaded %s\n", weight_path.c_str());

  const std::vector<std::uint8_t> blob = gdn::PackParameters(weights);
  const std::vector<ap_uint<128>> params = ToBeats(blob);
  const std::vector<float> side = gdn::BuildFp32Side(weights);
  std::printf("packed %zu words, %zu beats, side %zu floats\n",
              blob.size() / gdn::kPackedWordBytes, params.size(), side.size());

  gdn::RunState state;
  std::vector<float> logits(gdn::kVocabSize, 0.0f);

  // Drive both paths with the same varied token sequence so the recurrent state
  // actually moves; the demo checkpoint collapses to one token if we feed it
  // its own argmax.
  const int drive[] = {1,   9038, 2501, 263,  931,  29892, 297, 263,
                       2319, 4726, 4257, 4726, 29892, 10600, 1023, 1900};
  const int drive_count = static_cast<int>(sizeof(drive) / sizeof(drive[0]));

  int mismatches = 0;
  for (int step = 0; step < steps; ++step) {
    const int token = drive[step % drive_count];
    std::uint32_t fpga_next = 0;
    decode(token, step == 0 ? 1 : 0, params.data(), side.data(), &fpga_next);

    gdn::CpuForward(state, weights, token, logits.data());
    const int cpu_next = gdn::ArgmaxLogits(logits.data(), gdn::kVocabSize);

    const bool ok = static_cast<int>(fpga_next) == cpu_next;
    if (!ok) ++mismatches;
    std::printf("step %2d  token %5d  kernel %5u  cpu %5d  %s\n", step, token,
                fpga_next, cpu_next, ok ? "ok" : "MISMATCH");
  }

  std::printf("%s (%d mismatch of %d)\n", mismatches ? "FAILED" : "PASSED",
              mismatches, steps);
  return mismatches ? 1 : 0;
}
