# Figure 2 hardware architecture source

`draw_f2_c027.py` is the single editable source used for the final Figure 2
hardware architecture. It contains the frozen Stage-A geometry and exports the
same layout as neutral Stage A or README-styled Stage B.

## Reproduction

The module exposes these functions:

```python
from pathlib import Path
from draw_f2_c027 import render_matplotlib, render_svg, render_pptx

out = Path("figure2_outputs")
for stage in ("A", "B"):
    render_matplotlib(stage, out)
    render_svg(stage, out)
    render_pptx(stage, out)
```

Run from this directory with Python, Matplotlib, and `python-pptx` installed.
Matplotlib produces the paper PDF/PNG; the SVG is natively editable in a
browser; the PPTX contains native PowerPoint shapes and a native sampled
freeform loop. Use LibreOffice Impress for a real PPTX render and a browser for
the SVG render before publication.

## Geometry and style contract

- Stage A reproduces the historical `5180fc3` structure: Decode Kernel,
  Token Mixer, Channel Mixer, URAM/BRAM placement, module order, labels, and
  directed connections.
- Stage B freezes `GROUPS`, `BOXES`, `ARROWS`, and the four-segment Bézier loop,
  then applies only the palette/typography/line rules in `experiments/README.md`.
- The PPTX exporter removes theme style/effect references and samples the same
  Bézier loop into a native freeform so Impress and SVG/PDF share the visual
  curve. No raster assets or generated outputs are required inputs.

The source was derived from the paper-results Figure 2 gate records. Those
records and rendered outputs stay outside this repository; this PR publishes
only the editable source needed to regenerate them.
