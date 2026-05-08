"""TRACE-Kin v3 architecture diagram, Nature-style.

Two-column figure (180 mm wide). Outputs:
  figures/output/v3_architecture.pdf   ← vector, for the paper
  figures/output/v3_architecture.png   ← 600 dpi, for slides / previews

Run from the repo root:
  python figures/v3_architecture.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle


# ---------------------------------------------------------------------------
# Nature-style typography. Sans-serif, embedded fonts, 8 pt base.
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 0.8,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Palette ---------------------------------------------------------------
C_INPUT     = "#ECEEF1"
C_INPUT_ED  = "#7D7D7D"
C_GNN_FILL  = "#D8E5EE"
C_GNN_EDGE  = "#2E5C7E"
C_GNN_LITE  = "#F2F6F8"
C_RF_FILL   = "#F4E3C7"
C_RF_EDGE   = "#A86A1F"
C_GATE_FILL = "#E5E5E8"
C_GATE_EDGE = "#404040"
C_OUTPUT    = "#1A1A1A"
C_TEXT      = "#1A1A1A"
C_DIM       = "#5A5A5A"


def block(ax, x, y, w, h, *, fill, edge, lw=0.7, radius=0.04, zorder=2):
    """Plain rounded rectangle. Returns the patch so callers can ax-position children."""
    bx = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=zorder,
    )
    ax.add_patch(bx)
    return bx


def labeled_block(ax, x, y, w, h, title, *, fill, edge, subtitle=None,
                  title_color=C_TEXT, title_fontsize=8.5, title_weight="bold",
                  subtitle_fontsize=6.8, subtitle_color=C_DIM, lw=0.7, radius=0.04):
    """Block with a centered title and optional smaller subtitle below."""
    block(ax, x, y, w, h, fill=fill, edge=edge, lw=lw, radius=radius)
    cx = x + w / 2
    if subtitle:
        ax.text(cx, y + h * 0.62, title, ha="center", va="center",
                color=title_color, fontsize=title_fontsize, fontweight=title_weight,
                zorder=3)
        ax.text(cx, y + h * 0.30, subtitle, ha="center", va="center",
                color=subtitle_color, fontsize=subtitle_fontsize, style="italic",
                zorder=3)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                color=title_color, fontsize=title_fontsize, fontweight=title_weight,
                zorder=3)


def arrow(ax, x1, y1, x2, y2, *, color="#2A2A2A", lw=0.9,
          connectionstyle="arc3,rad=0", zorder=1):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", color=color, lw=lw,
        mutation_scale=8, connectionstyle=connectionstyle, zorder=zorder,
    )
    ax.add_patch(a)


def make_figure(out_pdf: Path, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Title strip ------------------------------------------------------
    ax.text(2, 61.2,
            "TRACE-Kin v3 — dual-head architecture with learned per-sample gating",
            fontsize=10.5, fontweight="bold", color=C_TEXT, ha="left")
    ax.text(2, 58.6,
            "v1's GNN backbone runs alongside an RF-style head; "
            "a sigmoid gate α selects per sample which path to trust.",
            fontsize=7.6, color=C_DIM, ha="left", style="italic")

    # ---- LEFT COLUMN: inputs ---------------------------------------------
    # Drug graph
    labeled_block(ax, 1.5, 47, 14.5, 7,
                  "Drug graph", subtitle="atoms · bonds · cliques",
                  fill=C_INPUT, edge=C_INPUT_ED)
    # Protein
    labeled_block(ax, 1.5, 38, 14.5, 7,
                  "Protein", subtitle="sequence + ESM (1280-d)",
                  fill=C_INPUT, edge=C_INPUT_ED)
    # ESM mean-pool (orange-tinted because it feeds RF + gate)
    labeled_block(ax, 1.5, 28, 14.5, 7,
                  "ESM mean-pool", subtitle="(B, 1280)",
                  fill=C_RF_FILL, edge=C_RF_EDGE)
    # Ligand fingerprints
    labeled_block(ax, 1.5, 18, 14.5, 7,
                  "Ligand FPs", subtitle="Morgan 2048 · MACCS 167",
                  fill=C_RF_FILL, edge=C_RF_EDGE)

    # ---- MIDDLE: GNN backbone (big container with internal stack) --------
    bb_x, bb_y, bb_w, bb_h = 22, 28, 28, 28
    block(ax, bb_x, bb_y, bb_w, bb_h, fill=C_GNN_LITE, edge=C_GNN_EDGE,
          lw=1.0, radius=0.05)
    ax.text(bb_x + bb_w / 2, bb_y + bb_h - 2.0,
            "v1 GNN backbone (verbatim)",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=C_GNN_EDGE, zorder=4)

    # Inner GNN components
    inner_x, inner_w = bb_x + 1.7, bb_w - 3.4
    labeled_block(ax, inner_x, bb_y + 18.6, inner_w, 4.3,
                  "PNAConv  ×3   (drug + protein)", subtitle=None,
                  fill=C_GNN_FILL, edge=C_GNN_EDGE,
                  title_fontsize=7.4, title_weight="normal", radius=0.05)
    labeled_block(ax, inner_x, bb_y + 13.6, inner_w, 4.3,
                  "MotifPool · MinCut pooling", subtitle=None,
                  fill=C_GNN_FILL, edge=C_GNN_EDGE,
                  title_fontsize=7.4, title_weight="normal", radius=0.05)
    labeled_block(ax, inner_x, bb_y + 8.6, inner_w, 4.3,
                  "DrugProteinConv  cross-attention", subtitle=None,
                  fill=C_GNN_FILL, edge=C_GNN_EDGE,
                  title_fontsize=7.4, title_weight="normal", radius=0.05)
    labeled_block(ax, inner_x, bb_y + 3.4, inner_w, 4.3,
                  "mol_pool (200) · prot_pool (200)", subtitle=None,
                  fill="#FFFFFF", edge=C_GNN_EDGE,
                  title_fontsize=7.4, title_weight="normal", radius=0.05)

    # Inputs → GNN backbone (drug + protein only)
    arrow(ax, 16, 50.5, bb_x, 50.5, lw=0.8, color="#444")
    arrow(ax, 16, 41.5, bb_x, 41.5, lw=0.8, color="#444")

    # ---- RIGHT COLUMN: heads + gate --------------------------------------
    # GNN regression head
    labeled_block(ax, 60, 49, 18, 8,
                  "GNN head", subtitle="MLP [400, 200, 1]",
                  fill=C_GNN_FILL, edge=C_GNN_EDGE, lw=0.9)
    # Gate (emphasized — bolder border)
    labeled_block(ax, 60, 36, 18, 8.5,
                  "Gate  α", subtitle="MLP [3895, 64, 1] · sigmoid",
                  fill=C_GATE_FILL, edge=C_GATE_EDGE, lw=1.2,
                  title_fontsize=9.5, radius=0.05)
    # RF-style head
    labeled_block(ax, 60, 14, 18, 8,
                  "RF-style head", subtitle="MLP [3495, 512, 128, 1]",
                  fill=C_RF_FILL, edge=C_RF_EDGE, lw=0.9)

    # Mol/prot pool (right edge of GNN backbone) → GNN head + gate
    arrow(ax, bb_x + bb_w, bb_y + 5.5, 60, 53, lw=0.85, color=C_GNN_EDGE,
          connectionstyle="arc3,rad=-0.20")
    ax.text(53.0, 49.5, "pred_gnn", fontsize=6.8, fontstyle="italic",
            color=C_GNN_EDGE, ha="center")

    arrow(ax, bb_x + bb_w, bb_y + 5.5, 60, 40.5, lw=0.7, color=C_GATE_EDGE,
          connectionstyle="arc3,rad=-0.10")

    # ESM mean → RF head + gate
    arrow(ax, 16, 31.5, 60, 19, lw=0.85, color=C_RF_EDGE,
          connectionstyle="arc3,rad=0.12")
    arrow(ax, 16, 31.5, 60, 39.5, lw=0.7, color=C_GATE_EDGE,
          connectionstyle="arc3,rad=-0.05")

    # Ligand FPs → RF head + gate
    arrow(ax, 16, 21.5, 60, 17, lw=0.85, color=C_RF_EDGE,
          connectionstyle="arc3,rad=0.05")
    arrow(ax, 16, 21.5, 60, 38, lw=0.7, color=C_GATE_EDGE,
          connectionstyle="arc3,rad=-0.10")

    # ---- Combiner + output -----------------------------------------------
    cx, cy = 86, 35.5
    ax.add_patch(Circle((cx, cy), 1.8, fill=False, edgecolor=C_OUTPUT,
                        linewidth=1.2, zorder=4))
    ax.text(cx, cy + 0.05, "+", ha="center", va="center",
            fontsize=12, fontweight="bold", color=C_OUTPUT, zorder=5)

    # GNN head → combiner with α label
    arrow(ax, 78, 53, cx - 1.4, cy + 1.4, lw=1.0, color=C_GNN_EDGE,
          connectionstyle="arc3,rad=-0.10")
    ax.text(82.4, 47.0, r"$\alpha$", fontsize=11, fontweight="bold",
            color=C_GNN_EDGE, ha="center")

    # RF head → combiner with (1-α) label
    arrow(ax, 78, 18, cx - 1.4, cy - 1.4, lw=1.0, color=C_RF_EDGE,
          connectionstyle="arc3,rad=0.10")
    ax.text(82.4, 24.0, r"$1-\alpha$", fontsize=10, fontweight="bold",
            color=C_RF_EDGE, ha="center")

    # Gate → combiner (controls weights)
    arrow(ax, 78, 40, cx - 1.7, cy + 0.4, lw=0.7, color=C_GATE_EDGE,
          connectionstyle="arc3,rad=-0.05")

    # Output box — clean ŷ only; the formula lives below the figure (full width).
    out_x, out_y, out_w, out_h = 91.5, 31.5, 7, 8
    block(ax, out_x, out_y, out_w, out_h, fill="#FFFFFF", edge=C_OUTPUT,
          lw=1.3, radius=0.06)
    ax.text(out_x + out_w / 2, out_y + out_h / 2, r"$\hat{y}$",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color=C_OUTPUT, zorder=3)

    # Combiner → output
    arrow(ax, cx + 1.8, cy, out_x, cy, lw=1.4, color=C_OUTPUT)

    # ---- Path labels next to RF head -------------------------------------
    ax.text(70.5, 12.0, "pred_rf", fontsize=6.8, fontstyle="italic",
            color=C_RF_EDGE, ha="center")

    # ---- Legend strip (bottom). Three labelled swatches; spaced to avoid
    # collisions on Arial wide enough to fit "catalytic kinetics" at 7 pt.
    leg_y = 5.5
    swatches = [
        (4,  C_GNN_FILL,  C_GNN_EDGE,  "GNN path"),
        (33, C_RF_FILL,   C_RF_EDGE,   "RF-style path"),
        (60, C_GATE_FILL, C_GATE_EDGE, "Learned gate"),
    ]
    captions = {
        "GNN path":      "graph-grounding; Ki / binding",
        "RF-style path": "closes the gap on catalytic kinetics",
        "Learned gate":  "per-sample routing",
    }
    for sx, fill, edge, label in swatches:
        ax.add_patch(Rectangle((sx, leg_y - 0.7), 3.6, 1.7,
                                facecolor=fill, edgecolor=edge, linewidth=0.6))
        ax.text(sx + 4.4, leg_y + 0.6, label,
                fontsize=7.2, fontweight="bold", va="center", color=C_TEXT)
        ax.text(sx + 4.4, leg_y - 0.7, captions[label],
                fontsize=6.5, va="center", color=C_DIM, style="italic")

    # Equation caption — full-width below the legend, like a Nature subfigure caption
    ax.text(50, 0.9,
            r"$\hat{y} \;=\; \alpha\;\hat{y}_{\mathrm{GNN}} \;+\; (1-\alpha)\;\hat{y}_{\mathrm{RF}}$",
            ha="center", va="center", fontsize=8.5, color=C_TEXT, zorder=3)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, format="png", bbox_inches="tight",
                pad_inches=0.05, dpi=600)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "output"
    out_pdf = out_dir / "v3_architecture.pdf"
    out_png = out_dir / "v3_architecture.png"
    make_figure(out_pdf, out_png)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
