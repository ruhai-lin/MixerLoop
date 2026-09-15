# 13M hardware implementation and validation record

This document records the binary contract, scheduler budget and validation evidence behind the [user manual](README.md). Results below describe the validated local build; raw generated reports under `outputs/` are not published release assets. Historical 15M results remain separate.

## Fixed profile

```text
part=xck26-sfvc784-2LV-c  clock=150 MHz  Vitis/Vivado=2025.2
dim=256  intermediate=704  layers=5  heads=6
head_k=32  head_v=64  conv_size=4  vocab=32000  context=1024
W8A8 symmetric INT8  group_size=32  scales/side/state=FP32
```

The software checkpoint is authoritative. The host rejects unsupported
geometry, norm epsilon, dtypes, metadata, truncated data and trailing bytes.
An explicit --loop_count override is a scheduling experiment: T1 on a trained
T4 checkpoint is not an independently trained GDN quality baseline.

```text
for physical layer:
  for loop:
    h_input = h
    h = h + GDN(RMSNorm(h))
    h = h + residual_weight[loop] * h_input
  h = h + FFN(FFNNorm(h))
```

## Quantization and binary format

Each matrix row is divided into consecutive groups of 32. Scale is
max(abs(w))/127 (zero group: scale=1); q=round(w/scale).clamp(-127,127).
This is the existing gdn.c / old MixerLoop Q8_0 algorithm, including
PyTorch ties-to-even weight rounding. Runtime activation quantization keeps
the old C/HLS ties-away rule; neither rule is silently changed.

### GDNe v3 binary contract

The 256-byte little-endian header contains the uint32 fields defined in
quantization.py:HEADER_FIELDS, followed by FP32 norm_eps, then zero padding.
Model IDs: GDN=1, MixerLoop=2. Dtype IDs: INT8=1, FP32=2.

tensor_specs defines payload order: one tied embedding/LM matrix, each
physical layer's Mixer and FFN tensors, final norm, then shared residual for
MixerLoop. GDN omits the residual payload; the host supplies zeros. Unused
loop slots are also zero. A_log is stored as -exp(A_log), as before.
Runtime S and convolution histories are not checkpoint tensors.

Each matrix stores all row-major INT8 values, then all row-major FP32 group
scales. Side tensors are contiguous FP32. There is no tensor padding or
scheduler-specific ordering. V3 adds explicit semantics, not a new quantizer.

weight.cpp packs device streams: 16 FP32 scales (one 512-bit word), then
eight row-pair INT8 words per 16-row/group block. Q/K/V/G are head-major; O
is input-group-major across output rows; W1/W3 are paired. Reordering device
streams does not require changing the public model format.

## Dataflow and memory budget

The DATAFLOW region has exactly nine processes: schedule, memory, scratch,
weight_router, q8, beta, conv, rec and post. All arithmetic is single-instance.
memory alone owns AXI; all AXI bundles connect to HP0. scratch alone owns the
SRAM weight storage. weight_router selects the source outside Q8.

Loop0 broadcasts all Mixer words to Q8 and SRAM as they arrive. Loops1–3
replay SRAM while HP0 prefetches FFN. Six heads advance through conv, beta,
recurrence and gated norm (in post). Each completed 64-value head supplies two
32-element O activation groups. The same Q8 engine accumulates all twelve
O groups; post fuses ordinary and MixerLoop residual writeback. Q8 processes
QKVG as one continuous packed stream; O group changes do not drain/restart
the MAC pipeline. Post quantizes each completed 32-value W1/W3 output group
while Q8 computes later rows, avoiding a separate full-FFN quantization tail.
Shared side parameters load once per layer, never during replay. The 4 KiB
global residual loads once per token, not once per layer.

S uses one 16-lane, two-pass engine and eight 64-bit URAM banks. Loop IDs
extend addresses only. Conv stores three previous taps for 192+192+384
channels per layer/loop slot. T4 does not clone arithmetic.

| Per-layer quantity | Value |
|---|---:|
| Q / K / V / G / O packed words | 864 / 864 / 1728 / 1728 / 1728 |
| Pinned Mixer | 6912 words |
| W1 / W3 / W2 | 3168 words each |
| FFN | 9504 words |
| Ideal FFN HP0 transfer | 38016 cycles |
| Pure FFN Q8 service | 9504 cycles |
| Ideal three-replay compute slack | 28512 cycles |
| Ideal cached Mixer budget | 9504 cycles/pass |
| Scratch / FFN ring | 16384 / 9472 words |

The ring has fixed read/write pointers and occupancy. Consumption immediately
frees an entry; the producer continues while Q8 waits for activations.
Full/empty boundaries do not rely on same-cycle URAM read/write to one
address. A newly arrived sole entry waits one iteration before consumption:
HLS otherwise schedules its write and the next iteration's read together.
Mixer weights remain pinned throughout the layer.

At C_M=9504, three replays allow 7128 prefetched FFN words. The measured T1
prefill peak adds 455 words while FFN activations become ready: 7583 is still
below ring capacity. At the measured C_M=8920, 455+3*8920/4=7145 matches RTL.
This service model is a budget, not proof of equal total latency: FFN
activation preparation, W1/W3→W2 dependency and actual stalls also matter.
LM head remains a common independent streaming tail.

## Numerical and cycle validation

Native simulation compares every token's argmax with CPU Q8 and checks reset
and loop switching. It links AMD's math simulation libraries: system libm is
not a bit-accurate substitute for the FPGA's exp/log/sqrt. The standalone CPU
host remains a scalar numerical reference, not a promise of bit-identical
generation: reduction order and a one-ULP scale difference can change an INT8
rounding decision. Use the AMD-math kernel simulation as the board oracle.
Native simulation is not a cycle model. RTL co-simulation uses the same
testbench and finite synthesized FIFOs; deadlock detection stays enabled.
The existing development entry, `scripts/build_sim.sh`, also rejects HLS
memory-dependence warnings, even if argmax agrees.
Run full-token RTL jobs sequentially: they take hours rather than seconds,
and a traced run can use over 20 GiB of RAM. Native gates cover reset/switching.
For cycle analysis, probe the generated simulator's command FIFO read/data,
weight FIFO read, head FIFO write and scratch occupancy signals under
/apatb_decode_top/AESL_inst_decode. Count a FIFO transfer only when its
read/empty_n or write/full_n handshake succeeds on a rising ap_clk edge.
Consecutive kPostAttn command reads bound a cached Mixer pass; kPostFfn
bounds the last pass. These probes do not add production hardware counters.

## Validation status

Canonical 13m is routed and running on KV260. Full-token T1 and T4 RTL
co-simulation both pass. Do not substitute historical 15M throughput or
timing for a 13m result.

- Target HF revision df35403838af4b557e3eb770068e2201f6d09c4c,
  mixerloop-13m/climbmix-10B-s1337.
- Software base commit 39e39b36393d79676f30995f441e2885df67e36a; training,
  configs and custom_models are unchanged by this hardware migration.
- Target binary: 14609136 bytes;
  SHA256 b19ddb1d01130c759b208304153f627385929896434ae1f97a21abfeab13c70e.
- Real GDN 13m reference: gdn-13m/fineweb-edu-10B-s1337 at the same HF revision;
  14605040 bytes, SHA256
  0ca6376152ac0e467ad86c1f273ed8eab7f43870325e60558642f99f9de7675a.
  Its geometry matches the kernel and the residual payload is absent/zero.
- Matrix-only PTQ max absolute error / MSE: MixerLoop 0.00154643 / 5.08337e-8;
  GDN 0.00138156 / 7.88539e-8. MixerLoop has 41 INT8 matrices and 52 FP32
  side tensors; GDN has 41 and 51. Tied LM weights are stored only once.
- 57 Python tests pass, including single/sharded GDN and MixerLoop PTQ.
- Native T1/T4: 16 tokens each, reset and cross-switch, no argmax mismatches.
  T2/T3 also pass four-token, reset and cross-switch checks with AMD math.
- Full-token T1/T4 RTL co-simulation: PASS, 945903/946599 cycles (one BOS
  transaction each, reset enabled), with no HLS memory-dependence warnings.
  The 696-cycle difference is 0.0736% of T1 latency. These are transaction
  latencies, not a measured multi-transaction initiation interval.
  Reports are saved as hardware/outputs/t1_cosim.rpt and t4_cosim.rpt.
  The recorded T4 launch shell failed after the tool's PASS because its
  script was edited during the run. Final script syntax, native rebuild
  and the memory-warning gate were checked separately; that shell exit
  is not counted as a clean end-to-end script run.
- Current csynth: 105105 LUT, 110039 FF, 437 DSP, 237 BRAM18, 64 URAM.
  All nine processes single-instance; state/scratch use 32/32 URAM.
- Synthesized RTL first-layer trace: cached Mixer 8920 cycles, including
  attention norm, activation quantization and both residuals. The breakdown is
  1560 preparation + 5184 QKVG + 97 Q8 phase change + 1728 O + 351 final tail.
  QKVG heads arrive every 864 cycles; O contributes 288 words per head without
  restarting the Q8 pipeline. Recurrence/post keep up with that schedule.
- First-layer T1/T4 intervals are 73751/74023 cycles: +272 cycles, not exactly
  zero. Three additional Mixer passes cost 26760 cycles but expose only 272
  at this boundary. Pure Q8 service is not the complete FFN latency budget.
  T4 ring peak is 7145/9472 words, with zero producer-full and consumer-empty
  stalls. Both runs receive the final FFN word at cycle 75567. These are RTL
  scheduler measurements, not board bandwidth or end-to-end throughput.
  Do not multiply the first-layer delta by five: the complete token exposes
  696 cycles, not 1360. Layer boundaries and inter-layer overlap matter.
- Post-route full design: 78527 LUT, 106490 FF, 453 DSP, 162 BRAM18-equivalent
  (77 RAMB36 + 8 RAMB18), 64 URAM. At 150 MHz, WNS +0.250 ns, TNS 0;
  hold WHS +0.010 ns, THS 0. No FP16 state conversion was needed.
  Xclbin SHA256:
  d6b7fd5d22182cdddbb079b18ac77339ec7a4e96fad96d909c4f192e022bb451.
- KV260, same MixerLoop weights/bitstream, 128 decode calls on an identical
  long prefix, seven alternating pairs after one warm-up pair:
  median T1 152.635 tok/s, T4 152.339 tok/s, ratio 99.806%.
  These host-wall-time measurements include normal per-token XRT overhead.
  A separate paired run using the real GDN/T1 and MixerLoop/T4 bundles at
  their default metadata gives 152.234/152.307 tok/s (measurement variation).
  The datasets differ, so this is a hardware test, not a model-quality ablation.
- Board MixerLoop T4: all 32 autoregressive token IDs from BOS + `Once`
  match native HLS with AMD math models. System-libm simulation first differs
  at decode 16; this was reproduced without changing the FPGA datapath.
  Real GDN T1 and the MixerLoop T1 override also match all 32 board token IDs.
  Reproduce native IDs with `kernel_sim WEIGHTS 32 LOOPS generate`: this mode
  uses BOS + `Once`, then the kernel's own predictions. It records output
  only; it is not a scalar-reference regression test.
- Known scalar-reference limit: forcing the T1-trained GDN checkpoint to T4
  produces one argmax difference in the four-token cross-switch test, also
  reproducible from a cold reset. The first differing INT8 activation crosses
  -74.5 because of a one-ULP scale difference. This off-metadata experiment is
  not counted as a passing regression or as a trained GDN baseline.
- Fixed 16-token HF FP32 vs CPU W8A8 sanity check: logits max absolute
  error 0.8647, mean absolute error 0.08164, argmax agreement 15/16,
  all finite. This includes dynamic activation quantization, unlike the
  weight-only PTQ error summary. Export CPU logits with GDN_SIM_LOGITS=PATH
  when running kernel_sim; its kDrive array specifies the input tokens.
  The same 16-token GDN 13m check gives max/mean error 0.97452/0.08486,
  argmax agreement 16/16 and no NaN/Inf.

Tensor PTQ error is not a model-quality score. Evaluate original HF quality
with eval/core_eval.py. PTQ evaluation requires the quantized inference path,
not an unchanged HF model with an adjacent binary.

## Historical 15M release

Commit 20fbcee preserves the board-validated GDNe v2 release: dim256, FFN768,
8 layers, 8 heads, K32/V32, no MixerLoop residual. Use that commit for old
weights; the current loader intentionally rejects them. Reports under
baselines/mixerloop_release/ are unchanged.

Old routed results: 74489 LUT, 98416 REG, 493 DSP, 104 BRAM tiles, 56 URAM;
WNS +0.082 ns, TNS 0 at 150 MHz. Steady medians: T1 119.549 tok/s, T4
118.484 tok/s (99.1%). Cached Mixer: 8738 cycles. These are architecture
reference numbers, not new 13m performance.

## Throughput measurement

Run on KV260 from the deployed bundle directory, with the ClimbMix Q8 file and bitstream identified above. The prompt repeats the same sentence 40 times, so the first 128 decode calls use a common teacher-forced input sequence. Each process starts with reset; the host timer begins after model/device setup and includes token processing and output. This is not a free-generation speed comparison.

```bash
PROMPT=$(printf 'Once upon a time there was a little village near a river. %.0s' {1..40})
for round in {0..7}; do
  if (( round % 2 == 0 )); then loops="1 4"; else loops="4 1"; fi
  for t in $loops; do
    echo "round=$round loops=$t"
    ./gdn_host -i "$PROMPT" -n 128 -t 0 --loop_count "$t"
  done
done
```

Discard round 0 and take each mode's median of rounds 1–7. Confirm every run reports 128 executed steps. Recorded host rates in tok/s:

| Round | T=1 | T=4 |
|---|---:|---:|
| 0 (warm-up) | 152.684 | 152.706 |
| 1 | 152.705 | 152.365 |
| 2 | 152.875 | 152.470 |
| 3 | 152.285 | 152.339 |
| 4 | 152.175 | 152.443 |
| 5 | 152.635 | 152.322 |
| 6 | 152.675 | 151.875 |
| 7 | 152.620 | 152.177 |

The original record is `hardware/outputs/board/board-throughput.json`.
Post-route source reports are:

- `hardware/outputs/link/reports/link/imp/impl_1_full_util_routed.rpt`
- `hardware/outputs/link/reports/link/imp/impl_1_MPSoC_ext_platform_wrapper_timing_summary_routed.rpt`

These paths are relative to the repository root and refer to generated local artifacts, not files included by a fresh clone. The compact measurements above preserve the reported results; rebuild to obtain a new complete report set. Rebuilding may change placement and bitstream hashes.
