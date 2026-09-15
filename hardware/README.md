# MixerLoop on KV260

We deployed 13M MixerLoop on KV260. The same accelerator runs one or four Mixer passes per layer, with measured T=4 throughput reaching 99.81% of T=1. Extra passes reuse on-chip Mixer weights and overlap with FFN weight transfers from DDR. The FFN runs once per layer.

The accelerator runs at 150 MHz with one shared 64-MAC INT8 engine, where MAC denotes a multiply-accumulate operation. All four passes share weights and compute units while maintaining independent recurrent states and convolution histories. Original Gated DeltaNet runs at T=1 with zero residual weights; MixerLoop runs at T=4 with trained `residual_weight`. Both use the same bitstream.

The deployment target is [MixerLoop 13M / ClimbMix 10B](https://huggingface.co/ruhai-lin/MixerLoop/tree/main/mixerloop-13m/climbmix-10B-s1337), using the Llama 2 32K tokenizer. Published unquantized HF evaluations report centered CORE v2 scores of [0.025990](https://huggingface.co/ruhai-lin/MixerLoop/blob/a8ca6f38787089857a3cecbc8c1c1142348364b8/mixerloop-13m/climbmix-10B-s1337/eval/core_eval.csv) for MixerLoop and [0.018149](https://huggingface.co/ruhai-lin/MixerLoop/blob/a8ca6f38787089857a3cecbc8c1c1142348364b8/gdn-13m/climbmix-10B-s1337/eval/core_eval.csv) for GDN with the same data budget and seed. These compare model quality. The throughput comparison below instead varies the runtime loop count of the same deployment weights. Formal CORE results for the quantized model have not yet been reported.

<!-- Architecture figure: to be supplied by the project author. -->
> Architecture figure placeholder: DDR → Mixer cache / FFN ring → shared Q8 engine; four persistent state slots; Mixer replay overlapping FFN prefetch.

## Quick start

The development machine needs Linux, Git, wget, CMake, a C++20 compiler, Vitis / Vivado 2025.2, and the `xilinx_kv260_base_202520_1` platform. Cross-compiling the ARM host requires an AArch64 SDK compatible with the board's XRT; this build was validated with the Xilinx common-image 2022.2 SDK. KV260 must be running Linux with XRT, `xmutil`, and SSH. See [AMD Vitis Getting Started](https://docs.amd.com/r/2025.2-English/Vitis-Tutorials-Getting-Started/Vitis-Introduction-and-Getting-Started) and the [KV260 User Guide](https://docs.amd.com/r/en-US/ug1089-kv260-starter-kit) for installation.

Check the tools in Bash on the development machine. Adjust the installation path as needed:

```bash
source /opt/xilinx/2025.2/Vitis/settings64.sh
v++ --version
vivado -version
g++ --version
cmake --version
```

### Get the model

Clone the repository. Run subsequent commands from its root unless a step explicitly runs on KV260.

```bash
git clone https://github.com/ruhai-lin/MixerLoop.git
cd MixerLoop
mkdir -p hardware/model
```

Public assets currently include the HF checkpoint, but not a standalone Q8 file or precompiled FPGA bundle. Download the validated HF revision and generate the deployment file with the PTQ exporter. PTQ converts trained weights without retraining or modifying the HF checkpoint. Use the Python environment from the [root environment guide](../README.md#environment).

```bash
source .venv/bin/activate
CHECKPOINT=outputs/mixerloop-13m/climbmix-10B-s1337
HF=https://huggingface.co/ruhai-lin/MixerLoop/resolve/df35403838af4b557e3eb770068e2201f6d09c4c/mixerloop-13m/climbmix-10B-s1337
mkdir -p "$CHECKPOINT"
wget -P "$CHECKPOINT" "$HF/config.json" "$HF/model.safetensors"
python hardware/quantization.py --checkpoint "$CHECKPOINT"
cp "$CHECKPOINT/climbmix-10B-s1337-q8.bin" hardware/model/

wget -O hardware/model/tokenizer.bin \
  https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
```

Quantization uses W8A8 with group size 32: matrix weights and runtime activations use INT8, with a scale for each group of 32 input values. Scales, side parameters, and persistent states remain FP32. The exporter supports single-file and standard sharded safetensors. It stops if the output exists; pass `--force` to overwrite it.

Verify that the deployment files match the validated artifacts:

```bash
echo 'b19ddb1d01130c759b208304153f627385929896434ae1f97a21abfeab13c70e  hardware/model/climbmix-10B-s1337-q8.bin' | sha256sum -c -
echo '50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361  hardware/model/tokenizer.bin' | sha256sum -c -
```

Run the CPU reference on the development machine to check model and tokenizer loading:

```bash
cmake -S hardware -B hardware/outputs/cpu
cmake --build hardware/outputs/cpu -j
hardware/outputs/cpu/gdn_host \
  --weight_path hardware/model/climbmix-10B-s1337-q8.bin \
  --vocab_path hardware/model/tokenizer.bin -i Once -n 32 -t 0
```

The program prints generated text, the number of executed decode steps, and throughput. The CPU reference checks deployment numerics, but its floating-point reduction order differs from the FPGA and does not guarantee identical generation for every input. Board comparisons use kernel simulation linked against AMD math models; see the [validation record](TECHNICAL.md#validation-status).

## Build from source

The following commands call HLS, the Vitis linker, and the cross-compiler directly. Generated files stay under the gitignored `hardware/outputs/` directory. HLS produces `decode.xo`; linking performs placement and routing and produces `binary_container_1.xclbin`; cross-compilation produces the board-side `gdn_host`.

### Synthesize the kernel

Create the HLS configuration and run synthesis from the repository root:

```bash
source /opt/xilinx/2025.2/Vitis/settings64.sh
PROJECT="$(pwd)/hardware"
mkdir -p hardware/outputs/hls/decode

cat > hardware/outputs/hls/decode/hls_config.cfg <<EOF
part=xck26-sfvc784-2LV-c

[hls]
flow_target=vitis
package.output.format=xo
package.output.syn=1
syn.top=decode
syn.file=$PROJECT/src/decode.cpp
syn.cflags=-I$PROJECT/src -DBUILD_DECODE_KERNEL
syn.interface.m_axi_max_widen_bitwidth=128
clock=150MHz
EOF

v++ -c --mode hls \
  --config hardware/outputs/hls/decode/hls_config.cfg \
  --work_dir hardware/outputs/hls/decode/work
```

### Link the bitstream

The connectivity below reproduces the bandwidth-limited memory configuration used in the reported experiment.

```bash
PLATFORM_ROOT="$XILINX_VITIS/base_platforms/xilinx_kv260_base_202520_1"
mkdir -p hardware/outputs/link hardware/outputs/logs/link

v++ -l -t hw \
  --platform "$PLATFORM_ROOT/xilinx_kv260_base_202520_1.xpfm" \
  hardware/outputs/hls/decode/work/decode.xo \
  -o hardware/outputs/link/binary_container_1.xclbin \
  --clock.default_freqhz 150000000 \
  --connectivity.sp decode_1.packed_params:HP0 \
  --connectivity.sp decode_1.side:HP0 \
  --connectivity.sp decode_1.next_token:HP0 \
  --save-temps \
  --temp_dir hardware/outputs/link/_x \
  --log_dir hardware/outputs/logs/link \
  --report_dir hardware/outputs/link/reports
```

### Compile the ARM host

Set `COMMON` to the installed common-image SDK directory. The subshell keeps its environment separate from Vitis. PIC/PIE is disabled to match the host's large code model.

```bash
COMMON=/path/to/xilinx-zynqmp-common-v2022.2
mkdir -p hardware/outputs/host
(
  unset LD_LIBRARY_PATH
  source "$COMMON/environment-setup-cortexa72-cortexa53-xilinx-linux"
  SYSROOT="$COMMON/sysroots/cortexa72-cortexa53-xilinx-linux"
  aarch64-xilinx-linux-g++ --version
  aarch64-xilinx-linux-g++ -Wall -Wextra -std=c++2a -O2 \
    -mcmodel=large -fno-PIC -fno-PIE -no-pie -g --sysroot="$SYSROOT" \
    -Ihardware/src -I"$SYSROOT/usr/include/xrt" \
    hardware/src/main.cpp hardware/src/decode.cpp \
    hardware/src/weight.cpp hardware/src/vocab.cpp \
    -L"$SYSROOT/usr/lib" -lxilinxopencl -lxrt_coreutil -lpthread -lrt -ldl \
    -o hardware/outputs/host/gdn_host
)
```

### Deploy and run

Assemble the runtime directory on the development machine. `binary_container_1.bin` is a copy of the xclbin, named for this platform's `xmutil` loading flow.

```bash
mkdir -p hardware/outputs/bundle/model
cp hardware/outputs/host/gdn_host hardware/outputs/bundle/
cp hardware/outputs/link/binary_container_1.xclbin hardware/outputs/bundle/binary_container_1.bin
cp "$PLATFORM_ROOT/sw/boot/pl.dtbo" hardware/outputs/bundle/
cp hardware/model/climbmix-10B-s1337-q8.bin hardware/model/tokenizer.bin hardware/outputs/bundle/model/
printf '{"shell_type":"XRT_FLAT","num_slots":"1"}\n' > hardware/outputs/bundle/shell.json

BOARD='ubuntu@<board-ip>'
ssh "$BOARD" 'mkdir -p ~/mixerloop13m'
scp -r hardware/outputs/bundle/. "$BOARD":~/mixerloop13m/
ssh "$BOARD"
```

Replace `<board-ip>` with your board's address. After connecting to KV260, run the following commands. Loading the application replaces the currently running FPGA logic.

```bash
cd ~/mixerloop13m
sudo mkdir -p /lib/firmware/xilinx/mixerloop13m
sudo cp binary_container_1.bin pl.dtbo shell.json /lib/firmware/xilinx/mixerloop13m/
sudo xmutil unloadapp
sudo xmutil loadapp mixerloop13m

./gdn_host -i "Once upon a time" -n 128 -t 0
./gdn_host -i "Once upon a time" -n 128 -t 0 --loop_count 1
```

The first command uses T=4 from the checkpoint metadata; the second runs the same weights at T=1. Each invocation resets state. The FPGA uses greedy argmax. `-n` limits decode calls, including prompt tokens, and an end token may stop generation early. These short-prompt commands test generation; they are not the fixed-input throughput benchmark below.

To restore the KV260 starter application after testing:

```bash
sudo xmutil unloadapp
sudo xmutil loadapp k26-starter-kits
```

## Performance and resources

The current 13M implementation completed placement and routing at 150 MHz on KV260 (`xck26-sfvc784-2LV-c`) and was validated on the board. The model has 5 physical layers, hidden size 256, 6 heads with K/V dimensions 32/64, and FFN intermediate size 704. Both settings below use the same bitstream.

| Board throughput | T=1 | T=4 |
|---|---:|---:|
| Median (tok/s) | 152.635 | 152.339 |
| Relative to T=1 | 100% | 99.81% |

The benchmark uses the same ClimbMix MixerLoop Q8 weights and a fixed long prefix, with 128 actual decode calls per run. Eight T=1/T=4 pairs alternate execution order; the first pair is discarded as warm-up, and each mode reports the median of the remaining seven runs. Timing includes per-token XRT calls and normal text output, but excludes model loading, parameter packing, and bitstream loading. The prefix covers all 128 inputs, so this measures token-by-token decode on fixed inputs, not 128 freely generated tokens. Commands and individual measurements are in the [throughput record](TECHNICAL.md#throughput-measurement).

FFN weight transfers hide most of the three additional Mixer passes, but total latency is not exactly equal. RTL measurements give 8,920 cycles for a complete cached Mixer pass and 945,903 / 946,599 cycles for one complete T=1/T=4 token transaction including reset. Board throughput includes host overhead and is reported separately from RTL cycle ratios.

| Post-route resource or timing metric | Value |
|---|---:|
| LUT | 78,527 |
| FF | 106,490 |
| DSP | 453 |
| BRAM18 equivalents | 162 |
| URAM | 64 |
| Operating frequency | 150 MHz |
| Setup WNS / TNS | +0.250 ns / 0 ns |
| Hold WHS / THS | +0.010 ns / 0 ns |

Resource counts cover the full design, including platform logic, from Vitis / Vivado 2025.2 post-route reports rather than HLS estimates. The 162 BRAM18 equivalents comprise 77 RAMB36 and 8 RAMB18 blocks. State and scratch each use 32 of the 64 URAM blocks. WNS/WHS denote worst setup/hold slack; TNS/THS denote total negative slack. Report paths, artifact hashes, and validation scope are recorded in the [technical report](TECHNICAL.md).

## Code layout

[decode.cpp](src/decode.cpp) implements the HLS kernel, fixed dataflow schedule, and CPU reference. [weight.cpp](src/weight.cpp) loads the model and packs device weights. [main.cpp](src/main.cpp) and [vocab.cpp](src/vocab.cpp) provide the host and tokenizer. [quantization.py](quantization.py) is the standalone local HF-to-Q8 exporter. [kernel_sim.cpp](tools/kernel_sim.cpp) checks token outputs, reset, and loop switching.

See the [technical report](TECHNICAL.md) for the binary format, cache budget, numerical validation limits, and historical 15M baseline. Training and software evaluation follow the [main repository workflow](../README.md); hardware export leaves training checkpoints unchanged.

## References and acknowledgments

[Swan](https://github.com/turingmotors/swan) provided a reference for C++ / HLS language-model deployment and KV260 project organization. [llama2.c](https://github.com/karpathy/llama2.c) provided a reference for C inference, the tokenizer binary format, and group-wise INT8 quantization.

The command-oriented structure of this guide draws on [llama2.hls](https://github.com/ruhai-lin/llama2.hls), [nanochat](https://github.com/karpathy/nanochat), and [AMD Vitis Getting Started](https://docs.amd.com/r/2025.2-English/Vitis-Tutorials-Getting-Started/Vitis-Introduction-and-Getting-Started). Refer to the [AMD KV260 User Guide](https://docs.amd.com/r/en-US/ug1089-kv260-starter-kit) for platform startup and board operations.
