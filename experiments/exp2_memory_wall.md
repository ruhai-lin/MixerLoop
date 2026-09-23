# Experiment 2: Climbing the Memory Wall

This experiment compares whole-decoder projection compute with active FFN weight traffic. It separates source-derived model coordinates from the illustrative MixerLoop scale/recurrence sweep; the latter is not a measured scaling result.

### Small-model architecture plane

Run from the repository root:

```bash
python3 experiments/exp2_memory_wall.py
```

The script writes `exp2_memory_wall.pdf`, `exp2_memory_wall.png`,
`model_architecture_metrics.csv` and `hardware_balance_metrics.csv` under
`experiments/outputs/`. The PDF is vector, 3.3 by 3.3 inches;
the PNG is 990 by 990 pixels. Reproduction needs only NumPy and Matplotlib.
Resolved model configurations and official source revisions are in the Python
registry; figure generation is offline and downloads no weights.

The main figure compares five small dense models, five small-active MoEs,
and the 13M / 100M / approximately 450M MixerLoop profiles. Earlier large-model
entries, including Ouro and LT2, remain source-pinned in the analysis CSV with
main_figure=False. They do not affect points, ellipses, viewport or legend.
There is no full-block or all-experts counterfactual arrow in this main figure.

### Counting contract

The horizontal coordinate is whole-decoder active FFN weight traffic, in
decimal GB per decoded token. The vertical coordinate is whole-decoder
Q/K/V/O projection compute (or MixerLoop Q/K/V/G/O), in GMAC per token.
Both coordinates describe the model under the same batch-one execution
assumptions, not a hardware coordinate or an advertised active parameter count.

The existing Q8 group-32 traffic factor is unchanged:
`b_w = 1 + 4/32 = 1.125 bytes/weight`.

```text
A_layer = 2 * d * (query_width + kv_width)
F_dense = 3 * L * d * intermediate_size          # SwiGLU / GeGLU
F_OPT   = 2 * L * d * intermediate_size          # two-matrix ReLU

F_MoE_layer = 3 * d * (top_k * expert_width + shared_count * shared_width)
F_token = sum(actual dense FFNs and active MoE FFNs over all layers)

D = b_w * F_token
x_plot = D / 1e9                                 [GB/token]
y_plot = sum(A_layer) / 1e9                       [GMAC/token]
```

Embedding, LM head, attention weights, norm, router and activation traffic are
excluded from x. Context-dependent QK^T/AV, FFN MACs and side operations are
excluded from y. OPT's biases do not add matrix MACs. Only routed-in experts
and always-active shared/dense FFNs contribute to MoE traffic.

### Reference models

All plotted coordinates are computed from the registry. The rounded user
reference coordinates are regression targets, not the source of x or y.

| Model | Active FFN weights (M) | FFN GB/token | Attention GMAC/token |
|---|---:|---:|---:|
| MiniMind2-104M | 75.497472 | 0.084934656 | 0.023592960 |
| OPT-125M | 56.623104 | 0.063700992 | 0.028311552 |
| SmolLM2-135M | 79.626240 | 0.089579520 | 0.026542080 |
| SmolLM2-360M | 235.929600 | 0.265420800 | 0.078643200 |
| Qwen2.5-0.5B | 313.786368 | 0.353009664 | 0.044040192 |
| FLAME-MoE-38M-100M | 9.701376 | 0.010914048 | 0.002359296 |
| TinyMoE-100M-2x8 | 17.694720 | 0.019906560 | 0.004423680 |
| TinyMoE-200M-2x16 | 21.233664 | 0.023887872 | 0.005308416 |
| Pluto-Nano-0.5 | 28.311552 | 0.031850496 | 0.006291456 |
| MiniMind-3-MoE | 44.826624 | 0.050429952 | 0.014155776 |

Nine entries use official published HF config.json revisions. FLAME releases
Megatron checkpoints rather than an HF config, so its geometry comes from the
official [38M release script](https://github.com/cmu-flame/FLAME-MoE/blob/e9b2fe2df3f1abb8dbb9ec0eabde8cdfb65e5c78/scripts/release/flame-moe-38m.sh)
and [model arguments](https://github.com/cmu-flame/FLAME-MoE/blob/e9b2fe2df3f1abb8dbb9ec0eabde8cdfb65e5c78/configs/model/flame-moe.sh).

FLAME has d=256, nine layers and 16 MHA heads. Its first layer has a dense
SwiGLU width of 1368; the remaining eight layers route to six of 64 experts,
each width 176, plus one shared FFN of width 352. Thus:

```text
F_FLAME = 3 * 256 * (1368 + 8 * (6 * 176 + 352)) = 9,701,376
```

This follows the executable release configuration. Its top-6 routed plus
two-expert-width shared branch differs from the HF model-card top-8 summary.
The discrepancy is recorded rather than inferred from the advertised 38M
active parameter count.

TinyMoE's null head_dim resolves to 384/8=48. Pluto's custom implementation
confirms three-matrix SwiGLU experts, top-1 routing, no shared expert and GQA
head width 64; training-only MTP is excluded. MiniMind-3-MoE's Qwen3-MoE export
sets eight MoE layers, expert width 2432, four experts and top-1 with no shared
expert. All relevant source URLs and revisions accompany the CSV.

### Regions and viewport

The main plot retains log x and log y. Its viewport is x=[0.002, 0.6] GB and
y=[0.00085, 0.8] GMAC. The small extension below 0.001 keeps the 13M annotation
clear of the axis.

Each category's ellipse is defined in z=(log10(x), log10(y)) space:

1. Center at the midpoint of the actual coordinate extrema.
2. Semiaxes equal half the log range plus log10(1.2), giving 20% multiplicative
   padding to each coordinate's range.
3. Uniformly expand if necessary to contain every observed point, then add
   a small log10(1.03) marker-edge clearance.
4. Sample the ellipse and exponentiate its vertices back into data space.

The regions are descriptive envelopes, not statistical confidence intervals.
Fills use pale terracotta (#F4EBE7) and mist blue (#EDF2F2), with separate thin
outlines (#B87563 and #7896A3). No convex hull, raw-data
Ellipse object, model-to-model connector, moved point or forced cluster
separation is used. Region bounds are printed on each run. Their size follows
the data, not the previous large-model viewport.

The bottom legend uses one centered rectangular frame with two rows. The first
row contains two MoE and three Dense entries; the second contains three MoE and
two Dense entries. Each of the five columns left-aligns its two markers and
text starts. Markers retain their category colors and individual shapes;
model names remain dark.

### MixerLoop and the unchanged Memory Wall

The current MixerLoop profile geometries and their trajectory coordinates are
unchanged:

```text
M = d * (2 * key_dim + 3 * value_dim)
F = 3 * d * intermediate_size
D_MixerLoop(T) = b_w * L * F
A_MixerLoop(T) = L * T * M
```

| Profile | FFN GB/token | Mixer GMAC/token, T=1 | Mixer GMAC/token, T=4 |
|---|---:|---:|---:|
| MixerLoop 13M | 0.003041280 | 0.001966080 | 0.007864320 |
| MixerLoop 100M | 0.049268736 | 0.031850496 | 0.127401984 |
| MixerLoop ~450M | 0.204374016 | 0.132120576 | 0.528482304 |

Only the 13M prototype retains the hollow T=1 circle, vertical arrow and filled
T=4 diamond. The larger profiles provide horizontal scale references, not
additional measurements. Three guides fan out from the 13M T=4 diamond.
The forest-green (#435E48) lower guide is solid, with small nodes at the
100M/450M traffic scales. The middle guide is translucent forest-green dashes,
slightly below the wall; the upper hypothetical guide uses sage gray (#BCC5BA).
They are schematic constrained/near-balanced/higher-budget scenarios, not a
fixed-T scaling law:

```text
u = log(D / D_13M) / log(D_450M / D_13M)
offset = log(A_13M_T4 / (wall_slope * D_13M))
A_center(D) = wall_slope * D * exp(offset * (1-u))
A_lower(D) = A_center(D) * exp(-1.2*u**2)
A_middle(D) = A_center(D) * exp(-0.18*u**2)
A_upper(D) = A_center(D) * exp(+1.2*u**2)
```

All three profiles have M/F=8/11. All guides retain the original 13M anchor,
slightly below the unchanged wall. The middle remains below it. Lower and upper
guides have equal opposite log-y offsets from the center guide: the lower bends
down while the upper bends up.
This is a visual scenario sweep, not a prediction that memory bandwidth changes
a fixed model's projection MAC count or that larger T improves quality.
The 100M/450M labels and nodes
mark profile traffic scales along the lower guide; exact T=1..4 model coordinates
remain in the table and CSV. Higher-budget scenarios are progressively lighter.
One curved arrow crosses the three guides to indicate increasing T; the
individual guides have no arrowheads. Guides stop before the upper plot border.
These guides make no model-quality claim.

The wall uses antique mustard (#C7A33A), with its label in #B28D24. Axes and
ordinary text use charcoal (#303330). The legend stays white, with a warm-gray
border (#B9B7B0) and text (#555753).

The existing Q8 traffic helper, P/B hardware calculation and principal wall
slope are unchanged. With attention/mixer-only y, the established
complete-compute balance deducts FFN arithmetic from the transfer window:

```text
I_req = P_MAC / B_memory
t_compute = (A_token + D/b_w) / P_MAC
t_transfer = D / B_memory
A_token = (I_req - 1/b_w) * D
```

The yellow KV260 1HP wall therefore retains slope 2.666692 MAC/B, from the same
I_req=3.555581 MAC/B. It is not fitted to the ten reference models. Higher
hardware requirements would shift a proportional line upward on log-log axes;
only the principal wall is shown.

### Reproducibility

Add a supported model dictionary entry with resolved architecture, layer count
and pinned source metadata. Main-figure membership defaults to True; the
archived large models explicitly set it to False. Point coordinates, category,
ellipse, grouped legend and CSV update automatically. No plotting formula or
manual final coordinate needs changing.

Checks retain the original projection/active-expert/MLA regressions and cover
two-matrix ReLU, dense prefixes, all ten rounded coordinate targets, full-block
and mixer-only recurrence accounting, fixed Q8 traffic and the unchanged wall.
The CSV retains all profiles, clearly flags main-figure membership, and records
whole-token FFN bytes and attention/mixer MACs. The raw throughput measurements,
hardware README and independent throughput figure are unchanged.

Suggested caption:

> **Climbing the Memory Wall.** Whole-decoder attention/mixer projection compute
> versus active FFN weight traffic at batch-one decode, with a common Q8
> representation. Small dense and small-active MoE references occupy distinct
> descriptive regions derived from their actual coordinates. The 13M green
> anchor shows T=1 to T=4 at fixed FFN traffic; the solid/dashed fan is an
> illustrative scale/recurrence-budget sweep, not measured scaling. The unchanged
> KV260 1HP Memory Wall accounts for FFN computation omitted from the ordinate.
> The comparison illustrates execution geometry, not quality equivalence or
> measured model throughput.
