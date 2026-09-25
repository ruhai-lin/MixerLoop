# Experiment 1: Memory bandwidth and recurrent compute

This experiment measures how memory bandwidth changes the cost of recurrent computation on the same accelerator. At one HP port, MixerLoop T=4 retains 99.91% of GDN T=1 throughput; at four ports, it retains 69.03%. The released implementation remains single-HP0. Follow the [hardware README](../hardware/README.md) for model downloads, tools, and the base build/deployment flow; the temporary changes and measurements are documented here.

The released source uses one 128-bit HP0 connection at 150 MHz. Its port ceiling is 2.4 GB/s (decimal); sustained bandwidth is measured separately below. The shared Q8 engine accepts one 512-bit packed word per cycle from on-chip memory, or 9.6 GB/s. A packed word can contain INT8 weights or FP32 scales. The balance ratio uses this weight path and excludes state and cache ports.

The bandwidth sweep uses 1, 2, and 4 HP ports without changing the Q8 engine, FP32 state, scratch capacity, or loop schedule. Four contiguous 128-bit parameter streams feed the existing memory process, which reassembles each 512-bit word. The same synthesized kernel is linked three ways:

| HP count | `params0` | `params1` | `params2` | `params3` | Port ceiling (GB/s) | On-chip / port-ceiling ratio |
|---:|---|---|---|---|---:|---:|
| 1 | HP0 | HP0 | HP0 | HP0 | 2.4 | 4 |
| 2 | HP0 | HP0 | HP3 | HP3 | 4.8 | 2 |
| 4 | HP0 | HP1 | HP2 | HP3 | 9.6 | 1 |

Side parameters and the output token remain on HP0. HP1 and HP2 share a DDR-controller interface, so the two-port point uses HP0 and HP3. Port count does not imply linear bandwidth scaling. The four-master one-port build must be measured too: it is not identical to the released single-master interface. See AMD's [HP-port topology](https://xilinx.github.io/Embedded-Design-Tutorials/docs/2021.2/build/html/docs/User_Guides/SPA-UG/docs/6-evaluating-high-performance-ports.html) and [kernel-port mapping](https://docs.amd.com/r/2025.2-English/ug1701-vitis-accelerated-embedded/Mapping-Kernel-Ports-to-Memory).

### Measured results

| HP ports | GDN T=1 (tok/s) | MixerLoop T=4 (tok/s) | T4 / T1 | GDN read traffic (GB/s) | MixerLoop read traffic (GB/s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 152.701 | 152.560 | 99.91% | 2.231 | 2.230 |
| 2 | 286.163 | 240.122 | 83.91% | 4.179 | 3.504 |
| 4 | 485.727 | 335.308 | 69.03% | 7.110 | 4.885 |

Extra Mixer work is hidden at the one-port point and becomes visible as bandwidth increases. Read traffic is measured by the DDR APM during decode. MixerLoop traffic falls at two and four ports when compute backpressure limits the consumer. Use HP count or the calibrated streaming bandwidth below for the first figure's shared x-axis.

These are medians of seven retained rounds after one warm-up pair, using the same models and 128-call prefix as the [release comparison](../hardware/README.md#performance-and-resources). Both models' 128 predicted tokens match across all three experimental images and the restored release image. For each experimental image, the full eight-pair run reads exactly 29,921,083,392 bytes. Per-token bytes observed on DDR slots 3/4/5 are respectively `14609904/0/0` (1 HP), `7374832/0/7235072` (2 HP), and `3757296/7235072/3617536` (4 HP), matching the intended striping. Idle traffic on these slots was zero. Linux used XRT 2.13.0 and the CPU remained at 1.333 GHz with the userspace governor. No DDR clock or QoS setting was changed between points.

The figure uses nominal port bandwidth (2.4, 4.8, and 9.6 GB/s) on the x-axis and measured throughput on the y-axis. Sustained streaming bandwidth is reported below.

### Streaming bandwidth calibration

The existing LM-head loop consumes one packed word per cycle without downstream stream backpressure. Its steady interior separates sustained parameter supply from complete-decoder compute gaps, using the same images and memory path.

| HP ports | LM stream (GB/s) | Minimum / maximum run (GB/s) | On-chip / measured stream ratio |
|---:|---:|---:|---:|
| 1 | 2.399983 | 2.399907 / 2.400189 | 4.000028 |
| 2 | 4.799728 | 4.799437 / 4.799798 | 2.000113 |
| 4 | 9.377286 | 9.373469 / 9.381770 | 1.023750 |

Each point is the median of six calibration runs: three per model, alternating order, with 128 calls per run. APM sampling targets 200 microseconds; actual retained intervals averaged 321–324 microseconds and bandwidth uses their measured durations. For each run, divide the sum of retained bytes by the sum of retained time. The sample standard deviations across runs are 0.000119, 0.000130, and 0.003047 GB/s, respectively.

To reproduce the window selection, accumulate unsigned read-byte deltas from slots 3/4/5, starting before the first decode. Every token reads exactly 14,609,904 bytes. Its final 9,216,000 bytes are the LM matrix. Keep an interval only if both endpoints belong to the same token and lie at least 1,048,576 bytes inside each end of the LM region. Include zero-byte intervals within that region so stalls are not discarded. This retained 6,332 / 2,767 / 1,110 intervals for 1/2/4 HP. Each six-run calibration read exactly 11,220,406,272 bytes, confirming the cumulative token boundaries; unrelated traffic on these APM slots would invalidate this method.

Single HP0 saturates during long streaming transfers while the whole-decoder average is lower. Four ports deliver about 97.68% of the nominal 9.6 GB/s ceiling for this platform, clock, host packing, and AXI master configuration.

### Routed resources

All three experimental images meet 150 MHz timing. Values below are full-design routed resources, including the platform. Each image contains the same single-instance nine-process HLS kernel; differences in LUT/FF come from interfaces, connectivity, and physical implementation.

| HP ports | LUT | FF | DSP | BRAM18 equivalents | URAM | WNS (ns) | WHS (ns) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 83,960 | 113,721 | 453 | 165 | 64 | +0.164 | +0.010 |
| 2 | 85,197 | 115,979 | 453 | 165 | 64 | +0.388 | +0.010 |
| 4 | 84,543 | 114,724 | 453 | 165 | 64 | +0.422 | +0.010 |

TNS and THS are zero for all three. BRAM is 79 RAMB36 plus 7 RAMB18 blocks. The smaller single-master release remains the published implementation; the experiment does not replace the [release resource table](../hardware/README.md#performance-and-resources).

<details>
<summary>Plot-ready measurements (CSV)</summary>

`host_sd_tok_s` is the sample standard deviation across seven retained runs. `event_ns` is the median of each run's mean OpenCL event duration over its 127 non-reset calls. APM values are medians of per-run byte/time ratios over 50 ms windows wholly within decode. Detailed per-run logs and routed reports are retained locally.

The plot-ready summary is stored in [data/exp1_bandwidth.csv](data/exp1_bandwidth.csv).

</details>

<details>
<summary>Build the bandwidth-sweep variant</summary>

To reproduce the bandwidth sweep, save the following block as `/tmp/bandwidth.diff` and apply it at the repository root with `git apply --unidiff-zero /tmp/bandwidth.diff`. It changes parameter transport, host buffers, and the matching simulation entry point. Public Q8 files and `PackParameters` stay unchanged. Every 64-byte packed word becomes four 16-byte lane entries; each lane buffer is 3,617,280 bytes. Each AXI master has four outstanding 256-beat reads, preserving the aggregate outstanding-data budget.

```diff
diff --git a/hardware/src/decode.cpp b/hardware/src/decode.cpp
index 4c5b3a3..09c7c84 100644
--- a/hardware/src/decode.cpp
+++ b/hardware/src/decode.cpp
@@ -335 +335,4 @@ static inline void unpack3(ConvWord word, float& a, float& b, float& c) {
-static inline ParameterWord read_word(const ParameterBeat* params, int offset) {
+static inline ParameterWord read_word(const ParameterBeat* params0,
+                                      const ParameterBeat* params1,
+                                      const ParameterBeat* params2,
+                                      const ParameterBeat* params3, int offset) {
@@ -337 +339,0 @@ static inline ParameterWord read_word(const ParameterBeat* params, int offset) {
-  const int beat = offset * kBeatsPerWord;
@@ -339,4 +341,4 @@ static inline ParameterWord read_word(const ParameterBeat* params, int offset) {
-  word.range(127, 0) = params[beat + 0];
-  word.range(255, 128) = params[beat + 1];
-  word.range(383, 256) = params[beat + 2];
-  word.range(511, 384) = params[beat + 3];
+  word.range(127, 0) = params0[offset];
+  word.range(255, 128) = params1[offset];
+  word.range(383, 256) = params2[offset];
+  word.range(511, 384) = params3[offset];
@@ -1047 +1049,3 @@ static void schedule_process(int token, int reset_state, int loop_count,
-static void axi_words(hls::stream<ParameterWord>& dst, const ParameterBeat* params,
+static void axi_words(hls::stream<ParameterWord>& dst,
+                      const ParameterBeat* params0, const ParameterBeat* params1,
+                      const ParameterBeat* params2, const ParameterBeat* params3,
@@ -1051,2 +1055,2 @@ static void axi_words(hls::stream<ParameterWord>& dst, const ParameterBeat* para
-#pragma HLS PIPELINE II = 4
-    dst.write(read_word(params, offset + it));
+#pragma HLS PIPELINE II = 1
+    dst.write(read_word(params0, params1, params2, params3, offset + it));
@@ -1058 +1062,3 @@ static void axi_broadcast(hls::stream<ParameterWord>& to_q8,
-                          const ParameterBeat* params, int offset, int count) {
+                          const ParameterBeat* params0, const ParameterBeat* params1,
+                          const ParameterBeat* params2, const ParameterBeat* params3,
+                          int offset, int count) {
@@ -1061,2 +1067,2 @@ static void axi_broadcast(hls::stream<ParameterWord>& to_q8,
-#pragma HLS PIPELINE II = 4
-    const ParameterWord word = read_word(params, offset + it);
+#pragma HLS PIPELINE II = 1
+    const ParameterWord word = read_word(params0, params1, params2, params3, offset + it);
@@ -1068 +1074,3 @@ static void axi_broadcast(hls::stream<ParameterWord>& to_q8,
-static void load_embedding(hls::stream<float>& out, const ParameterBeat* params,
+static void load_embedding(hls::stream<float>& out,
+                           const ParameterBeat* params0, const ParameterBeat* params1,
+                           const ParameterBeat* params2, const ParameterBeat* params3,
@@ -1078 +1086 @@ static void load_embedding(hls::stream<float>& out, const ParameterBeat* params,
-        read_word(params, base + group * kWordsPerGroup);
+        read_word(params0, params1, params2, params3, base + group * kWordsPerGroup);
@@ -1080 +1088 @@ static void load_embedding(hls::stream<float>& out, const ParameterBeat* params,
-        read_word(params, base + group * kWordsPerGroup + 1 + pair);
+        read_word(params0, params1, params2, params3, base + group * kWordsPerGroup + 1 + pair);
@@ -1091 +1099,3 @@ static void load_embedding(hls::stream<float>& out, const ParameterBeat* params,
-static void memory_process(const ParameterBeat* params, const float* side,
+static void memory_process(const ParameterBeat* params0, const ParameterBeat* params1,
+                           const ParameterBeat* params2, const ParameterBeat* params3,
+                           const float* side,
@@ -1109 +1119 @@ static void memory_process(const ParameterBeat* params, const float* side,
-      load_embedding(embed, params, cmd.token);
+      load_embedding(embed, params0, params1, params2, params3, cmd.token);
@@ -1123 +1133 @@ static void memory_process(const ParameterBeat* params, const float* side,
-      axi_broadcast(axi_out, fill_words, params, PackedQOffset(layer, 0),
+      axi_broadcast(axi_out, fill_words, params0, params1, params2, params3, PackedQOffset(layer, 0),
@@ -1126 +1136 @@ static void memory_process(const ParameterBeat* params, const float* side,
-      axi_words(fill_words, params, PackedW13Offset(layer), kFfnPackedWords);
+      axi_words(fill_words, params0, params1, params2, params3, PackedW13Offset(layer), kFfnPackedWords);
@@ -1128 +1138 @@ static void memory_process(const ParameterBeat* params, const float* side,
-      axi_words(axi_out, params, kPackedTokOffset, kPackedTokWords);
+      axi_words(axi_out, params0, params1, params2, params3, kPackedTokOffset, kPackedTokWords);
@@ -1618 +1628,4 @@ void decode(int token, int reset_state, int loop_count,
-            const ParameterBeat* __restrict packed_params, const float* side,
+            const ParameterBeat* __restrict params0,
+            const ParameterBeat* __restrict params1,
+            const ParameterBeat* __restrict params2,
+            const ParameterBeat* __restrict params3, const float* side,
@@ -1620,3 +1633,8 @@ void decode(int token, int reset_state, int loop_count,
-#pragma HLS INTERFACE m_axi port = packed_params bundle = params0 \
-    depth = kPackedTotalBeats max_read_burst_length = 256 \
-    num_read_outstanding = 16
+#pragma HLS INTERFACE m_axi port = params0 bundle = params0 \
+    depth = kPackedTotalBeats / 4 max_read_burst_length = 256 num_read_outstanding = 4
+#pragma HLS INTERFACE m_axi port = params1 bundle = params1 \
+    depth = kPackedTotalBeats / 4 max_read_burst_length = 256 num_read_outstanding = 4
+#pragma HLS INTERFACE m_axi port = params2 bundle = params2 \
+    depth = kPackedTotalBeats / 4 max_read_burst_length = 256 num_read_outstanding = 4
+#pragma HLS INTERFACE m_axi port = params3 bundle = params3 \
+    depth = kPackedTotalBeats / 4 max_read_burst_length = 256 num_read_outstanding = 4
@@ -1631 +1649,4 @@ void decode(int token, int reset_state, int loop_count,
-#pragma HLS INTERFACE s_axilite port = packed_params bundle = control
+#pragma HLS INTERFACE s_axilite port = params0 bundle = control
+#pragma HLS INTERFACE s_axilite port = params1 bundle = control
+#pragma HLS INTERFACE s_axilite port = params2 bundle = control
+#pragma HLS INTERFACE s_axilite port = params3 bundle = control
@@ -1734 +1755 @@ void decode(int token, int reset_state, int loop_count,
-  memory_process(packed_params, side, next_token, mem_cmds, embed, rms_final,
+  memory_process(params0, params1, params2, params3, side, next_token, mem_cmds, embed, rms_final,
@@ -1757 +1778 @@ void decode(int token, int reset_state, int loop_count,
-    memory_process(packed_params, side, next_token, mem_cmds, embed, rms_final,
+    memory_process(params0, params1, params2, params3, side, next_token, mem_cmds, embed, rms_final,
@@ -2034 +2055,2 @@ int Decode(int token, bool reset_state, int loop_count, cl::CommandQueue& q,
-  err = q.enqueueTask(kernel);
+  cl::Event event;
+  err = q.enqueueTask(kernel, nullptr, &event);
@@ -2040,0 +2063,7 @@ int Decode(int token, bool reset_state, int loop_count, cl::CommandQueue& q,
+  if (std::getenv("MIXERLOOP_PROFILE")) {
+    const auto begin = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
+    const auto end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
+    std::fprintf(stderr, "kernel_ns=%llu reset=%d next=%u\n",
+                 static_cast<unsigned long long>(end - begin), reset_state,
+                 *next_token);
+  }
diff --git a/hardware/src/main.cpp b/hardware/src/main.cpp
index 38cf4d0..dd45e58 100644
--- a/hardware/src/main.cpp
+++ b/hardware/src/main.cpp
@@ -237 +237 @@ int main(int argc, char** argv) {
-    // Match llama2.hls M5.1: one canonical parameter blob on one HP port.
+    // Stripe each packed word across four contiguous 128-bit streams.
@@ -241,2 +241,8 @@ int main(int argc, char** argv) {
-    AlignedVector<std::uint8_t> packed(packed_src.size());
-    std::memcpy(packed.data(), packed_src.data(), packed_src.size());
+    AlignedVector<std::uint8_t> packed[4];
+    for (int p = 0; p < 4; ++p) {
+      packed[p].resize(packed_src.size() / 4);
+      for (std::size_t i = 0; i < packed_src.size() / 64; ++i) {
+        std::memcpy(packed[p].data() + 16 * i,
+                    packed_src.data() + 64 * i + 16 * p, 16);
+      }
+    }
@@ -294,3 +300,7 @@ int main(int argc, char** argv) {
-    cl::Buffer buffer_params(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
-                             packed.size(), packed.data(), &err);
-    OCL_THROW_IF_ERROR(err, "buffer_params");
+    cl::Buffer buffer_params[4];
+    for (int p = 0; p < 4; ++p) {
+      buffer_params[p] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
+                                   packed[p].size(), packed[p].data(), &err);
+      OCL_THROW_IF_ERROR(err, "buffer_params");
+      OCL_CHECK(err, err = kernel.setArg(3 + p, buffer_params[p]));
+    }
@@ -303,3 +313,2 @@ int main(int argc, char** argv) {
-    OCL_CHECK(err, err = kernel.setArg(3, buffer_params));
-    OCL_CHECK(err, err = kernel.setArg(4, buffer_side));
-    OCL_CHECK(err, err = kernel.setArg(5, buffer_next));
+    OCL_CHECK(err, err = kernel.setArg(7, buffer_side));
+    OCL_CHECK(err, err = kernel.setArg(8, buffer_next));
@@ -307 +316,2 @@ int main(int argc, char** argv) {
-                       {buffer_params, buffer_side}, 0));
+                       {buffer_params[0], buffer_params[1], buffer_params[2],
+                        buffer_params[3], buffer_side}, 0));
@@ -347,0 +358,7 @@ int main(int argc, char** argv) {
+    if (std::getenv("MIXERLOOP_PROFILE")) {
+      std::cerr << "decode_begin_ns="
+                << std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count()
+                << " decode_end_ns="
+                << std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count()
+                << '\n';
+    }
diff --git a/hardware/tools/kernel_sim.cpp b/hardware/tools/kernel_sim.cpp
index 242cf7d..c9b199c 100644
--- a/hardware/tools/kernel_sim.cpp
+++ b/hardware/tools/kernel_sim.cpp
@@ -11,0 +12,2 @@
+#include <array>
+#include <stdexcept>
@@ -22 +24,3 @@ extern "C" void decode(int token, int reset_state, int loop_count,
-                       const ap_uint<128>* packed_params, const float* side,
+                       const ap_uint<128>* params0, const ap_uint<128>* params1,
+                       const ap_uint<128>* params2, const ap_uint<128>* params3,
+                       const float* side,
@@ -31,6 +35,14 @@ const int kDriveCount = static_cast<int>(sizeof(kDrive) / sizeof(kDrive[0]));
-std::vector<ap_uint<128>> ToBeats(const std::vector<std::uint8_t>& bytes) {
-  std::vector<ap_uint<128>> beats(bytes.size() / 16);
-  for (std::size_t i = 0; i < beats.size(); ++i) {
-    ap_uint<128> value = 0;
-    for (int b = 0; b < 16; ++b) {
-      value.range(b * 8 + 7, b * 8) = bytes[i * 16 + b];
+using Lanes = std::array<std::vector<ap_uint<128>>, 4>;
+
+Lanes ToBeats(const std::vector<std::uint8_t>& bytes) {
+  Lanes beats;
+  for (int p = 0; p < 4; ++p) {
+    beats[p].resize(bytes.size() / 64);
+    for (std::size_t i = 0; i < beats[p].size(); ++i) {
+      for (int b = 0; b < 16; ++b) {
+        beats[p][i].range(b * 8 + 7, b * 8) = bytes[i * 64 + p * 16 + b];
+        if (beats[p][i].range(b * 8 + 7, b * 8).to_uint() !=
+            bytes[i * 64 + p * 16 + b]) {
+          throw std::runtime_error("stripe roundtrip mismatch");
+        }
+      }
@@ -38 +49,0 @@ std::vector<ap_uint<128>> ToBeats(const std::vector<std::uint8_t>& bytes) {
-    beats[i] = value;
@@ -44 +55 @@ int RunTokens(const char* label, int loop_count, int steps, bool reset_first,
-              const std::vector<ap_uint<128>>& params, const std::vector<float>& side,
+              const Lanes& params, const std::vector<float>& side,
@@ -63 +74,2 @@ int RunTokens(const char* label, int loop_count, int steps, bool reset_first,
-    decode(token, reset, loop_count, params.data(), side.data(), &fpga_next);
+    decode(token, reset, loop_count, params[0].data(), params[1].data(),
+           params[2].data(), params[3].data(), side.data(), &fpga_next);
@@ -124 +136 @@ int main(int argc, char** argv) {
-  const std::vector<ap_uint<128>> params = ToBeats(blob);
+  const Lanes params = ToBeats(blob);
@@ -127 +139 @@ int main(int argc, char** argv) {
-              blob.size() / gdn::kPackedWordBytes, params.size(), side.size(),
+              blob.size() / gdn::kPackedWordBytes, params[0].size() * 4, side.size(),
```

Build and run the existing kernel simulation for both checkpoints. Synthesize this kernel once, then link the resulting XO three times using the port table above. Replace the original `decode_1.packed_params:HP0` linker option with four `--connectivity.sp decode_1.paramsN:HPx` options; retain the side/output mappings, platform, and 150 MHz clock. Keep separate output directories so no validated release artifact is overwritten. Recompile the ARM host because the kernel arguments changed. Check post-route timing and resources for every image before loading it through the same `xmutil` flow.

Set `MIXERLOOP_PROFILE=1` for the temporary host to print per-token OpenCL event nanoseconds, predicted token IDs, and the host decode interval. For each image, run the two real models in alternating order for eight rounds, discard the first pair, and retain all seven remaining observations. Use the same 128-call teacher-forced prefix: repeat `Once upon a time there was a little village near a river. ` forty times. Compare predicted token IDs across images, not just printed text: teacher forcing prints the supplied prefix even if predictions are wrong.

After measurements, run `git apply --reverse --unidiff-zero /tmp/bandwidth.diff` and reload the released single-HP0 image. No multi-port kernel or alternate source tree is part of the release.

</details>

### Data definitions

Keep host throughput, device execution time, and DDR traffic separate. Host throughput includes XRT and output overhead. OpenCL event duration measures each kernel invocation; summarize non-reset calls separately from the first call. DDR APM read-byte counters measure traffic on controller slots 3, 4, and 5; slot 4 includes both HP1 and HP2 and must only be counted once. Sample the 32-bit counters every 50 ms and compute unsigned differences to handle wraparound. Record idle traffic and use only sampling windows wholly inside the decode interval, excluding model loading and buffer migration. Application traffic is not an independent measurement of maximum available DDR bandwidth: compute backpressure can reduce it.

The Linux [APM driver](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18842046/APM) exposes the DDR monitor at physical address `0xfd0b0000` through UIO; identify it by its sysfs address rather than assuming a UIO number. In advanced mode, select Read Byte Count (metric 3) for slots 3/4/5 in counters 0/1/2: the low 24 bits of selector register `0x44` are `0xa38363`. Enable metric counters with bit 0 of control register `0x300`, then read counters at `0x100`, `0x110`, and `0x120`. Preserve and restore the original selector/control values, and refuse to replace an already active measurement. These definitions follow AMD's [AXI performance-monitor driver](https://github.com/Xilinx/embeddedsw/tree/master/XilinxProcessorIPLib/drivers/axipmon/src). Slots 3 and 5 also see DisplayPort and FPD DMA traffic, respectively; a nonzero idle baseline must be investigated before attributing their bytes to this accelerator.

The following constants and baseline measurements make the arithmetic reproducible. A `word` is 512 bits including scale overhead. Service cycles assume an uninterrupted one-word-per-cycle Q8 input; they are not complete scheduler latency.

```csv
quantity,value,unit,kind
clock,150000000,Hz,nominal
packed_word,512,bit,configuration
hp_port,128,bit,configuration
q8_stream_rate,1,word/cycle,achieved_II
int8_mac_lanes,64,MAC/cycle,configuration
weight_bits,8,bit,configuration
activation_bits,8,bit,configuration
scale_bits,32,bit,configuration
state_bits,32,bit,configuration
quant_group,32,values,configuration
layers,5,count,configuration
mixer_weights_per_layer,6912,word,layout
ffn_weights_per_layer,9504,word,layout
lm_head_weights,144000,word,layout
matrix_reads_per_token,226080,word,layout
q8_service_words_t4,329760,word,layout_including_replay
matrix_MACs_t1,12861440,MAC/token,layout
matrix_MACs_t4,18759680,MAC/token,layout
embedding_reads_per_token,16,word,layout
side_reads_per_token,34940,float32,layout
output_per_token,4,byte,layout
logical_transfer_per_token,14609908,byte,layout
scratch,16384,word,configuration
pinned_mixer,6912,word,configuration
ffn_ring,9472,word,configuration
cached_mixer,8920,cycle,release_RTL_trace
ffn_q8_service,9504,cycle,service_floor
ffn_transfer_1hp,38016,cycle,ideal_port_budget
extra_mixer_passes,3,count,configuration
extra_mixer_compute_per_layer,26760,cycle,derived_from_RTL
extra_mixer_compute_per_token,133800,cycle,derived_from_RTL
ffn_memory_slack_1hp,28512,cycle,ideal_service_budget
ring_peak_1hp,7145,word,release_RTL_trace
gdn_t1_token_with_reset,945903,cycle,release_RTL_trace
mixerloop_t4_token_with_reset,946599,cycle,release_RTL_trace
```

The last two rows are single-token RTL measurements including reset. The release trace's ring peak includes 455 words prefetched before the three replay passes. Its cached Mixer latency includes normalization, projections, recurrence, and residual writeback. One MAC means one multiply-accumulate (two arithmetic operations). Each group uses eight data words and one scale word, so the 64-MAC engine's sustained packed-service ceiling is `64*8/9` MAC/cycle. Matrix MAC counts exclude FP32 side operations and recurrent-state updates.


## Reproduce the figure

Run from the repository root:

```bash
python3 experiments/exp1_bandwidth.py
```

Outputs: `outputs/exp1_bandwidth.pdf` and `outputs/exp1_bandwidth.png`, relative to this directory. The script reads [data/exp1_bandwidth.csv](data/exp1_bandwidth.csv). This is the existing six-row summary of retained measurements, not per-run raw samples; detailed logs remain local. Error bars show the observed minimum and maximum, not confidence intervals.

### Retained latency diagnostic

The earlier exposed-latency statistic remains available as a diagnostic,
not the architecture figure's ordinate:

```text
D_extra = 5 * 3 * 8920 / 150 MHz = 892 microseconds
rho_r = (L_MixerLoop,r - L_GDN,r) / D_extra
rho = median(rho_r), over retained paired rounds r=1,...,7
```

Each L is the run's mean OpenCL event duration over 127 non-reset calls.
Pair runs before taking the median; the denominator is from the release RTL
trace, not a new per-HP trace.

```csv
hp,stream_to_DDR_ratio,extra_latency_us,rho,rho_min,rho_max
1,4.000028,0.321031,0.000360,-0.027345,0.042149
2,2.000113,656.103543,0.735542,0.701148,0.768737
4,1.023750,903.457543,1.012845,1.000806,1.033985
```
