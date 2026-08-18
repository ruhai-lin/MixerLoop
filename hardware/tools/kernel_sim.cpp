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
#include <string>
#include <vector>

extern "C" void decode(int token, int reset_state, int loop_count,
                       const ap_uint<128>* packed_params, const float* side,
                       std::uint32_t* next_token, std::uint32_t* stats);

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

int RunCompare(const char* label, int loop_count, int steps, bool reset_first,
               const std::vector<ap_uint<128>>& params, const std::vector<float>& side,
               const gdn::Weights& weights, std::uint32_t* ring_max_out) {
  gdn::RunState state;
  std::vector<float> logits(gdn::kVocabSize, 0.0f);
  int mismatches = 0;
  std::uint32_t ring_max = 0;
  for (int step = 0; step < steps; ++step) {
    const int token = kDrive[step % kDriveCount];
    std::uint32_t fpga_next = 0;
    std::uint32_t stats[8] = {};
    const int reset = (reset_first && step == 0) ? 1 : 0;
    decode(token, reset, loop_count, params.data(), side.data(), &fpga_next,
           stats);
    if (reset) {
      state = gdn::RunState();
    }
    gdn::CpuForward(state, weights, token, logits.data(), loop_count);
    const int cpu_next = gdn::ArgmaxLogits(logits.data(), gdn::kVocabSize);
    const bool ok = static_cast<int>(fpga_next) == cpu_next;
    if (!ok) ++mismatches;
    if (stats[0] > ring_max) {
      ring_max = stats[0];
    }
    if (loop_count == gdn::kMaxLoopCount && stats[0] > 6528u) {
      ++mismatches;
      std::printf("%s step %2d  ring_max %u exceeds 6528\n", label, step,
                  stats[0]);
    }
    std::printf("%s step %2d  token %5d  kernel %5u  cpu %5d  ring_max %u  %s\n",
                label, step, token, fpga_next, cpu_next, stats[0],
                ok ? "ok" : "MISMATCH");
    std::fflush(stdout);
  }
  if (ring_max_out) {
    *ring_max_out = ring_max;
  }
  return mismatches;
}

} // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const std::string weight_path =
      argc > 1 ? argv[1] : "model/climbmix15M_demo_q8.bin";
  const int steps = argc > 2 ? std::atoi(argv[2]) : 8;
  const bool run_gates = argc > 4 && std::string(argv[4]) == "gates";

  gdn::Weights weights;
  gdn::LoadWeights(weights, weight_path);
  std::printf("loaded %s (header loop_count=%d)\n", weight_path.c_str(),
              weights.loop_count);

  int loop_count = weights.loop_count;
  if (argc > 3) {
    loop_count = std::atoi(argv[3]);
    if (weights.loop_count > 0 && loop_count != weights.loop_count) {
      std::fprintf(stderr,
                   "loop_count mismatch: header %d vs argument %d\n",
                   weights.loop_count, loop_count);
      return 2;
    }
  }
  if (loop_count < 1) {
    std::fprintf(stderr,
                 "loop_count unspecified: pass it as argv[3] or store T in the "
                 "checkpoint header pad\n");
    return 2;
  }
  if (loop_count > gdn::kMaxLoopCount) {
    loop_count = gdn::kMaxLoopCount;
  }

  const std::vector<std::uint8_t> blob = gdn::PackParameters(weights);
  const std::vector<ap_uint<128>> params = ToBeats(blob);
  const std::vector<float> side = gdn::BuildFp32Side(weights);
  std::printf("packed %zu words, %zu beats, side %zu floats, loop_count=%d\n",
              blob.size() / gdn::kPackedWordBytes, params.size(), side.size(),
              loop_count);
  std::fflush(stdout);

  std::uint32_t ring_max = 0;
  int mismatches =
      RunCompare("main", loop_count, steps, true, params, side, weights,
                 &ring_max);

  if (run_gates) {
    std::uint32_t gate_ring = 0;
    mismatches += RunCompare("reset", loop_count, 4, true, params, side, weights,
                             &gate_ring);
    if (gate_ring > ring_max) {
      ring_max = gate_ring;
    }
    const int other = loop_count == gdn::kMaxLoopCount ? 1 : gdn::kMaxLoopCount;
    mismatches += RunCompare("cross", other, 4, true, params, side, weights,
                             &gate_ring);
    if (gate_ring > ring_max) {
      ring_max = gate_ring;
    }
    mismatches += RunCompare("back", loop_count, 4, true, params, side, weights,
                             &gate_ring);
    if (gate_ring > ring_max) {
      ring_max = gate_ring;
    }
  }

  std::printf("%s (%d mismatch of %d) ring_max=%u\n",
              mismatches ? "FAILED" : "PASSED", mismatches, steps, ring_max);
  return mismatches ? 1 : 0;
}
