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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

extern "C" void decode(int token, int reset_state, int loop_count,
                       const ap_uint<128>* packed_params, const float* side,
                       std::uint32_t* next_token);

namespace {

const int kDrive[] = {1,    9038, 2501, 263,  931,  29892, 297, 263,
                      2319, 4726, 4257, 4726, 29892, 10600, 1023, 1900};
const int kDriveCount = static_cast<int>(sizeof(kDrive) / sizeof(kDrive[0]));

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

int RunTokens(const char* label, int loop_count, int steps, bool reset_first,
              const std::vector<ap_uint<128>>& params, const std::vector<float>& side,
              const gdn::Weights& weights, bool generate = false) {
  gdn::RunState state;
  std::vector<float> logits(gdn::kVocabSize, 0.0f);
  int mismatches = 0;
  // Optional row-major FP32 CPU logits for comparison with the HF model.
  std::ofstream dump;
  const char* dump_path = std::getenv("GDN_SIM_LOGITS");
  if (!generate && dump_path && std::strcmp(label, "main") == 0) {
    dump.exceptions(std::ios::failbit | std::ios::badbit);
    dump.open(dump_path, std::ios::binary);
  }
  int next = 1;
  for (int step = 0; step < steps; ++step) {
    // Generation uses the fixed BOS + "Once" prefix, then the kernel's tokens.
    const int token = generate ? (step < 2 ? kDrive[step] : next)
                               : kDrive[step % kDriveCount];
    std::uint32_t fpga_next = 0;
    const int reset = (reset_first && step == 0) ? 1 : 0;
    decode(token, reset, loop_count, params.data(), side.data(), &fpga_next);
    next = static_cast<int>(fpga_next);
    if (generate) {
      std::printf("step %2d  token %5d  kernel %5u\n", step, token, fpga_next);
      continue;
    }
    if (reset) {
      state = gdn::RunState();
    }
    gdn::CpuForward(state, weights, token, logits.data(), loop_count);
    if (dump.is_open()) {
      dump.write(reinterpret_cast<const char*>(logits.data()),
                 logits.size() * sizeof(float));
    }
    const int cpu_next = gdn::ArgmaxLogits(logits.data(), gdn::kVocabSize);
    const bool ok = static_cast<int>(fpga_next) == cpu_next;
    if (!ok) ++mismatches;
    std::printf("%s step %2d  token %5d  kernel %5u  cpu %5d  %s\n",
                label, step, token, fpga_next, cpu_next,
                ok ? "ok" : "MISMATCH");
    std::fflush(stdout);
  }
  return mismatches;
}

} // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  // RTL co-simulation launches the testbench without argv; build_sim.sh
  // passes the same inputs through the environment in that case.
  const char* env_weight = std::getenv("GDN_SIM_WEIGHT");
  const char* env_steps = std::getenv("GDN_SIM_STEPS");
  const char* env_loops = std::getenv("GDN_SIM_LOOPS");
  const std::string weight_path =
      argc > 1 ? argv[1] : env_weight ? env_weight : "";
  const int steps = argc > 2 ? std::atoi(argv[2]) : env_steps ? std::atoi(env_steps) : 16;
  const bool run_gates = (argc > 4 && std::string(argv[4]) == "gates") ||
                         std::getenv("GDN_SIM_GATES");
  const bool generate = (argc > 4 && std::string(argv[4]) == "generate") ||
                         std::getenv("GDN_SIM_GENERATE");
  if (weight_path.empty() || steps < 1) {
    std::fprintf(stderr, "usage: kernel_sim WEIGHTS [STEPS] [LOOPS] [gates|generate]\n");
    return 2;
  }

  gdn::Weights weights;
  gdn::LoadWeights(weights, weight_path);
  std::printf("loaded %s (header loop_count=%d)\n", weight_path.c_str(),
              weights.loop_count);

  int loop_count = weights.loop_count;
  if (argc > 3 || env_loops) {
    loop_count = std::atoi(argc > 3 ? argv[3] : env_loops);
  }
  if (loop_count < 1 || loop_count > gdn::kMaxLoopCount) {
    std::fprintf(stderr, "loop_count must be in 1..4\n");
    return 2;
  }

  const std::vector<std::uint8_t> blob = gdn::PackParameters(weights);
  const std::vector<ap_uint<128>> params = ToBeats(blob);
  const std::vector<float> side = gdn::BuildFp32Side(weights);
  std::printf("packed %zu words, %zu beats, side %zu floats, loop_count=%d\n",
              blob.size() / gdn::kPackedWordBytes, params.size(), side.size(),
              loop_count);
  std::fflush(stdout);

  int mismatches = RunTokens("main", loop_count, steps, true, params, side,
                            weights, generate);

  if (run_gates && !generate) {
    const int gate_steps = steps < 4 ? steps : 4;
    mismatches +=
        RunTokens("reset", loop_count, gate_steps, true, params, side, weights);
    const int other = loop_count == gdn::kMaxLoopCount ? 1 : gdn::kMaxLoopCount;
    mismatches += RunTokens("cross", other, gate_steps, true, params, side, weights);
    mismatches +=
        RunTokens("back", loop_count, gate_steps, true, params, side, weights);
  }

  if (generate) {
    std::printf("GENERATED %d tokens (no scalar-reference comparison)\n", steps);
  } else {
    std::printf("%s (%d mismatch)\n", mismatches ? "FAILED" : "PASSED", mismatches);
  }
  return mismatches ? 1 : 0;
}
