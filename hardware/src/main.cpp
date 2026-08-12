#include "decode.hpp"
#include "vocab.hpp"
#include "weight.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef USE_CPU_ONLY
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/opencl.hpp>

#define OCL_CHECK(error, call)                                             \
  call;                                                                    \
  if (error != CL_SUCCESS) {                                               \
    std::cerr << __FILE__ << ":" << __LINE__ << " OpenCL error " << error  \
              << " calling " #call << std::endl;                           \
    std::exit(EXIT_FAILURE);                                               \
  }

#define OCL_THROW_IF_ERROR(error, what)                                    \
  if (error != CL_SUCCESS) {                                               \
    throw std::runtime_error(std::string(what) +                           \
                             " failed with OpenCL error " +                \
                             std::to_string(error));                       \
  }

template <typename T>
struct aligned_allocator {
  using value_type = T;
  T* allocate(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) {
      throw std::bad_alloc();
    }
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t) { free(p); }
};

template <typename T>
using AlignedVector = std::vector<T, aligned_allocator<T>>;
#endif // USE_CPU_ONLY

#ifdef USE_CPU_ONLY
struct Sampler {
  float temperature = 0.0f;
  float topp = 0.9f;
  unsigned long long rng_state = 1;
};

static unsigned int RandomU32(unsigned long long& state) {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return static_cast<unsigned int>((state * 0x2545F4914F6CDD1Dull) >> 32);
}

static float RandomF32(unsigned long long& state) {
  return static_cast<float>(RandomU32(state) >> 8) / 16777216.0f;
}

static int SampleToken(Sampler& sampler, std::vector<float>& logits) {
  if (sampler.temperature <= 1e-6f) {
    return gdn::ArgmaxLogits(logits.data(), static_cast<int>(logits.size()));
  }
  for (float& v : logits) v /= sampler.temperature;
  const float max_val = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0f;
  for (float& v : logits) {
    v = std::exp(v - max_val);
    sum += v;
  }
  for (float& v : logits) v /= sum;

  const float coin = RandomF32(sampler.rng_state);
  if (sampler.topp <= 0.0f || sampler.topp >= 1.0f) {
    float cdf = 0.0f;
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
      cdf += logits[i];
      if (coin < cdf) return i;
    }
    return static_cast<int>(logits.size()) - 1;
  }

  std::vector<std::pair<float, int>> candidates;
  const float cutoff =
      (1.0f - sampler.topp) / static_cast<float>(logits.size() - 1);
  for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
    if (logits[i] >= cutoff) candidates.push_back({logits[i], i});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  float cumulative = 0.0f;
  std::size_t last = candidates.empty() ? 0 : candidates.size() - 1;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    cumulative += candidates[i].first;
    if (cumulative > sampler.topp) {
      last = i;
      break;
    }
  }
  const float target = coin * cumulative;
  float cdf = 0.0f;
  for (std::size_t i = 0; i <= last; ++i) {
    cdf += candidates[i].first;
    if (target < cdf) return candidates[i].second;
  }
  return candidates[last].second;
}
#endif // USE_CPU_ONLY

struct Args {
  std::string weight_path = "./model/climbmix15M_demo_q8.bin";
  std::string vocab_path = "./model/tokenizer.bin";
  std::string xclbin_path = "./binary_container_1.bin";
  std::string prompt = "";
  int max_seq = 128;
  float temp = 0.0f;
  float topp = 0.9f;
  unsigned long long seed = 1337;
  bool help = false;
};

static void ParseArgs(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    const std::string opt = argv[i];
    auto require_value = [&](const char* name) -> char* {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (opt == "--weight_path") {
      args.weight_path = require_value("--weight_path");
    } else if (opt == "--vocab_path") {
      args.vocab_path = require_value("--vocab_path");
    } else if (opt == "--xclbin") {
      args.xclbin_path = require_value("--xclbin");
    } else if (opt == "--prompt" || opt == "-i") {
      args.prompt = require_value(opt.c_str());
    } else if (opt == "--max_seq" || opt == "-n") {
      args.max_seq = std::stoi(require_value(opt.c_str()));
    } else if (opt == "--temp" || opt == "-t") {
      args.temp = std::stof(require_value(opt.c_str()));
    } else if (opt == "--topp" || opt == "-p") {
      args.topp = std::stof(require_value(opt.c_str()));
    } else if (opt == "--seed" || opt == "-s") {
      args.seed = std::stoull(require_value(opt.c_str()));
    } else if (opt == "--help" || opt == "-h") {
      args.help = true;
    } else {
      throw std::runtime_error("unknown option: " + opt);
    }
  }
}

static void PrintUsage(const char* exe) {
  std::cout << "Usage: " << exe << " [options]\n"
            << "  --weight_path PATH   GDN Q8 checkpoint, default "
               "./model/climbmix15M_demo_q8.bin\n"
            << "  --vocab_path PATH    tokenizer.bin, default ./model/tokenizer.bin\n"
            << "  --xclbin PATH        XRT binary, default ./binary_container_1.bin\n"
            << "  -i, --prompt TEXT    Prompt text\n"
            << "  -n, --max_seq N      Decode steps, default 128\n"
            << "  -t, --temp FLOAT     Temperature, 0 means argmax\n"
            << "  -p, --topp FLOAT     Top-p threshold, default 0.9\n"
            << "  -s, --seed INT       RNG seed\n";
}

int main(int argc, char** argv) {
  try {
    Args args;
    ParseArgs(argc, argv, args);
    if (args.help) {
      PrintUsage(argv[0]);
      return 0;
    }

    std::cout << "GDN 15M constants\n"
              << "  dim       : " << gdn::kDim << "\n"
              << "  hidden_dim: " << gdn::kHiddenDim << "\n"
              << "  n_layers  : " << gdn::kNumLayers << "\n"
              << "  n_heads   : " << gdn::kNumHeads << "\n"
              << "  head dims : " << gdn::kHeadKDim << "/" << gdn::kHeadVDim << "\n"
              << "  vocab_size: " << gdn::kVocabSize << "\n"
              << "  seq_len   : " << gdn::kSeqLen << std::endl;

    gdn::Weights weights;
    gdn::LoadWeights(weights, args.weight_path);
    gdn::Tokenizer tokenizer =
        gdn::LoadTokenizer(args.vocab_path, gdn::kVocabSize);
    std::vector<int> prompt_tokens =
        gdn::Encode(tokenizer, args.prompt, true, false);
    if (prompt_tokens.empty()) {
      prompt_tokens.push_back(1);
    }

#ifdef USE_CPU_ONLY
    gdn::RunState state;
    std::vector<float> logits(gdn::kVocabSize, 0.0f);
    Sampler sampler{args.temp, args.topp, args.seed};
#else
    if (args.temp >= 1e-5f) {
      throw std::runtime_error(
          "FPGA build returns exact argmax only; use --temp 0 for now");
    }

    // Match llama2.hls M5.1: one canonical parameter blob on one HP port.
    std::vector<std::uint8_t> packed_src = gdn::PackParameters(weights);
    std::vector<float> side_src = gdn::BuildFp32Side(weights);

    AlignedVector<std::uint8_t> packed(packed_src.size());
    std::memcpy(packed.data(), packed_src.data(), packed_src.size());
    AlignedVector<float> side(side_src.size());
    std::memcpy(side.data(), side_src.data(), side_src.size() * sizeof(float));
    AlignedVector<std::uint32_t> next_aligned(1, 0);

    cl_int err = CL_SUCCESS;
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Platform::get(&platforms);
    bool found_device = false;
    for (const auto& platform : platforms) {
      if (platform.getInfo<CL_PLATFORM_NAME>() == "Xilinx") {
        platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
        found_device = !devices.empty();
        if (found_device) break;
      }
    }
    if (!found_device) {
      throw std::runtime_error("unable to find Xilinx accelerator device");
    }

    std::ifstream bin_file(args.xclbin_path, std::ios::binary);
    if (!bin_file) {
      throw std::runtime_error("xclbin not available: " + args.xclbin_path);
    }
    bin_file.seekg(0, std::ios::end);
    const unsigned nb = static_cast<unsigned>(bin_file.tellg());
    bin_file.seekg(0, std::ios::beg);
    std::vector<char> binary(nb);
    bin_file.read(binary.data(), nb);

    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;
    bool programmed = false;
    for (const auto& device : devices) {
      OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
      OCL_CHECK(err, queue = cl::CommandQueue(context, device,
                                              CL_QUEUE_PROFILING_ENABLE, &err));
      cl::Program::Binaries bins{{binary.data(), nb}};
      program = cl::Program(context, {device}, bins, nullptr, &err);
      if (err == CL_SUCCESS) {
        OCL_CHECK(err, kernel = cl::Kernel(program, "decode", &err));
        programmed = true;
        break;
      }
    }
    if (!programmed) {
      throw std::runtime_error("failed to program any Xilinx device");
    }

    cl::Buffer buffer_params(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                             packed.size(), packed.data(), &err);
    OCL_THROW_IF_ERROR(err, "buffer_params");
    cl::Buffer buffer_side(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                           side.size() * sizeof(float), side.data(), &err);
    OCL_THROW_IF_ERROR(err, "buffer_side");
    cl::Buffer buffer_next(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                           sizeof(std::uint32_t), next_aligned.data(), &err);
    OCL_THROW_IF_ERROR(err, "buffer_next");

    OCL_CHECK(err, err = kernel.setArg(2, buffer_params));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_side));
    OCL_CHECK(err, err = kernel.setArg(4, buffer_next));
    OCL_CHECK(err, err = queue.enqueueMigrateMemObjects(
                       {buffer_params, buffer_side}, 0));
    OCL_CHECK(err, err = queue.finish());
#endif // USE_CPU_ONLY

    int token = prompt_tokens[0];
    int next = token;
    const auto start = std::chrono::steady_clock::now();

    for (int pos = 0; pos < args.max_seq; ++pos) {
#ifdef USE_CPU_ONLY
      gdn::CpuForward(state, weights, token, logits.data());
#else
      const int fpga_next = gdn::Decode(token, pos == 0, queue, kernel,
                                        next_aligned.data(), buffer_next);
      if (fpga_next < 0) {
        throw std::runtime_error("decode kernel failed");
      }
#endif

      if (pos + 1 < static_cast<int>(prompt_tokens.size())) {
        next = prompt_tokens[pos + 1];
      } else {
#ifdef USE_CPU_ONLY
        next = SampleToken(sampler, logits);
#else
        next = fpga_next;
#endif
      }
      if (next == 2) break;

      const std::string piece = gdn::DecodePiece(tokenizer, token, next);
      std::cout.write(piece.data(), static_cast<std::streamsize>(piece.size()));
      std::cout << std::flush;
      token = next;
    }
    std::cout << "\n";

    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    std::cout << "Time : " << seconds << "[s]\n"
              << "Speed: " << args.max_seq / seconds << "[tok/s]" << std::endl;
    std::cout.flush();
    std::exit(EXIT_SUCCESS);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
