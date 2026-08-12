#ifndef GDN_DECODE_HPP_
#define GDN_DECODE_HPP_

#include "config.hpp"

#include <cstdint>

#ifndef BUILD_DECODE_KERNEL
#include <vector>

namespace gdn {

struct Weights;

struct RunState {
  std::vector<float> x, xb, hb, hb2;
  std::vector<float> q, k, v, gate, beta, decay, linear_out;
  std::vector<float> q_conv_state, k_conv_state, v_conv_state, S;
  std::vector<std::int8_t> xq_q;
  std::vector<float> xq_s;

  RunState();
};

void CpuForward(RunState& state, const Weights& weights, int token,
                float* logits);
int ArgmaxLogits(const float* logits, int count);

} // namespace gdn
#endif

#if !defined(USE_CPU_ONLY) && !defined(BUILD_DECODE_KERNEL)
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/opencl.hpp>
#endif // FPGA host build

#if !defined(USE_CPU_ONLY) && !defined(BUILD_DECODE_KERNEL)
namespace gdn {

int Decode(int token, bool reset_state, cl::CommandQueue& q,
           cl::Kernel& kernel, std::uint32_t* next_token,
           cl::Buffer& next_token_buffer);

} // namespace gdn
#endif // FPGA host build

#endif // GDN_DECODE_HPP_
