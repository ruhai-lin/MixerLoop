# MixerLoop single-HP release baseline

Frozen from the release build on 2026-08-24 with Vitis/Vivado 2025.2,
`xck26-sfvc784-2LV-c`, and a 150 MHz target clock.

The archived reports establish three release contracts:

- one instance of every DATAFLOW process and one shared Q8 engine;
- routed timing closure at +0.082 ns WNS with 74,489 LUT, 493 DSP, 104 BRAM,
  and 56 URAM in the kernel;
- KV260 steady-state medians of 119.549 tok/s for T=1 and 118.484 tok/s for
  T=4, or 99.1% throughput retention on the same single-HP bitstream.

`decode_csynth.rpt` is the HLS synthesis report. The two routed reports come
from the Vitis platform link. `kv260_board.txt` records the raw board runs used
for the reported medians.
