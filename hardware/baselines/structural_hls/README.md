# Structural HLS baseline

Frozen 2026-08-18 from the first canonical DATAFLOW kernel in
`hardware/src/decode.cpp`. Later MixerLoop work (runtime `T`, loop slots,
FFN ring, overlap) must keep this 9-process graph.

## Process graph

```
controller ×1
memory_process ×1      HP0 owner
scratch_process ×1     mixer URAM [0, 5759]
weight_router ×1       AXI vs SRAM mux
q8_process ×1          unified weight_stream, consume II=1
beta_process ×1
conv_process ×1
rec_process ×1         packed S URAM inside this process
post_process ×1
```

## kernel-sim

```
model: hardware/model/tinystories15m_t1_q8.bin
steps: 16
result: PASSED (0 mismatch of 16)
```

See `kernel_sim_16.txt`.

## csynth (Vitis 2025.2, xck26-sfvc784-2LV-c, 150 MHz)

| Resource | Used | Available | Util |
|---|---:|---:|---:|
| LUT | 81913 | 117120 | 69% |
| FF | 75383 | 234240 | 32% |
| DSP | 292 | 1248 | 23% |
| BRAM_18K | 178 | 288 | 61% |
| URAM | 32 | 64 | 50% |
| Estimated clock | 5.172 ns | target 6.67 ns | |

Compute instances (each ×1):

| Process | LUT | DSP | URAM |
|---|---:|---:|---:|
| q8_process | 7231 | 57 | 0 |
| rec_process | 22790 | 121 | 8 |
| post_process | 21392 | 45 | 0 |
| conv_process | 9581 | 35 | 0 |
| beta_process | 7447 | 30 | 0 |
| memory_process | 7907 | 4 | 0 |
| scratch_process | 613 | 0 | 24 |
| weight_router | 402 | 0 | 0 |
| controller | 382 | 0 | 0 |

Q8 consume pipeline II=1. AXI feeder II=4. Scratch read/write II=1.

Full reports: `decode_csynth.rpt`, `csynth.rpt`.
