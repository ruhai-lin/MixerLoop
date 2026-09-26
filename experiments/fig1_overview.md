# Figure 1: Architecture and performance

The left panel preserves the author's draw.io architecture. The right panel plots
ARC-E, ARC-C, HellaSwag, PIQA, WinoGrande, and BoolQ as six groups of four bars.
The legend identifies the model and parameter count. Bars within each task touch.

## Inputs

- `data/f1_software_main_arch.drawio`: editable architecture source.
- `data/f1_software_main_arch.pdf`: the author's vector export, used for composition.
- `RESULTS` in `fig1_performance.py`: the six task scores supplied on 2026-09-25.

GDN 400M uses the updated scores 59.80, 28.58, 40.46, 65.94, 51.46, and 60.03.
Training budgets remain in the script: 15B tokens for GDN 400M and 100B for the
other three rows. They are omitted from the figure. The script checks six-task
averages against the supplied table; the chart displays individual task scores.

FullLoop's six-task mean is distinct from LT2's reported eight-task average of
55.74. The author has flagged the GDN source rows for verification; update their
six scores and expected averages together when that verification is complete.
MixerLoop uses 100B FineWeb-Edu tokens. No cross-model corpus equivalence is
assumed here.

## Reproduce

Requires Matplotlib.

```bash
python3 experiments/fig1_performance.py
```

The script writes `outputs/f1_performance.pdf` and its 300-dpi PNG preview.
The panel uses a 3.6-inch square canvas, cropped to its content with a 0.02-inch
margin and embedded TrueType fonts. The legend sits below the task labels.
The vertical axis is `Score (%)` and starts at zero.

## Compose in LaTeX

Place the architecture export and generated performance PDF in the paper's
`figures/` directory. Both panels retain vector content. The architecture can
be edited in draw.io independently of the performance script. The original
source files remain unchanged.

```latex
% Preamble: \usepackage{graphicx,xcolor}
% \definecolor{minggreen}{HTML}{3AA278}
\begin{figure}[t]
  \centering
  \begin{minipage}[t]{0.45\linewidth}
    \centering
    {\sffamily\bfseries\color{minggreen}(a) Architecture\par}
    \smallskip
    \includegraphics[width=\linewidth]{figures/f1_software_main_arch.pdf}
  \end{minipage}\hfill
  \begin{minipage}[t]{0.53\linewidth}
    \centering
    {\sffamily\bfseries\color{minggreen}(b) Zero-shot Performance\par}
    \smallskip
    \includegraphics[width=\linewidth]{figures/f1_performance.pdf}
  \end{minipage}
  \caption{MixerLoop architecture and six-task zero-shot performance.
  The legend identifies model parameter counts.}
  \label{fig:overview}
\end{figure}
```

The performance panel uses the shared palette: MixerLoop `#3AA278`, GDN 1.3B
`#D75C5D`, GDN 400M `#E79596`, and FullLoop `#6667AB`. Original
architecture colors and typography remain intact.
