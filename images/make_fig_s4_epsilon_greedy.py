#!/usr/bin/env python3
"""Figure 4: the epsilon-greedy decision policy of the proposed eps-L-SPBO.

Panel (a) is the two-stage decision tree applied to every decision variable of
the average-student group; panel (b) is the resulting marginal share of variable
updates at the parameter values used in the paper.

Palette: categorical slots 1 (blue) and 2 (orange) of the validated reference
palette, plus a neutral for the unchanged-SPBO branch. Verified with
validate_palette.js: all six checks PASS (worst adjacent CVD dE 24.7).

Usage: python3 make_fig_s4_epsilon_greedy.py
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- parameters of the enhancement (Table 1 of the paper) -------------------
EPSILON = 0.1
P_THRESHOLD = 0.6

SHARE_EXPLOIT = (1 - EPSILON) * 100                 # 90 %
SHARE_SPREAD = EPSILON * P_THRESHOLD * 100          # 6 %
SHARE_LEVY = EPSILON * (1 - P_THRESHOLD) * 100      # 4 %

# --- design tokens ----------------------------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"          # text-primary
INK_MUTED = "#52514e"    # text-secondary
GRID = "#d9d8d4"
NEUTRAL = "#8a8984"      # the unchanged-SPBO branch: deliberately recessive
BLUE = "#2a78d6"         # categorical slot 1 -> Levy jump
ORANGE = "#eb6834"       # categorical slot 2 -> uniform spread

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,   # embed TrueType: MDPI requires editable/embedded fonts
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(7.2, 2.85))
gs = fig.add_gridspec(1, 2, width_ratios=[1.52, 1.0], wspace=0.30,
                      left=0.008, right=0.978, top=0.845, bottom=0.145)

# ===========================================================================
# Panel (a): two-stage decision tree
# ===========================================================================
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def node(x, y, w, h, title, subtitle, edge, face, bold_title=True):
    """Rounded node box with a title line and a smaller condition line."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=2.4",
        linewidth=1.0, edgecolor=edge, facecolor=face,
        mutation_aspect=0.55, zorder=3))
    if subtitle:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                fontsize=7.6, color=INK,
                fontweight="bold" if bold_title else "normal", zorder=4)
        ax.text(x + w / 2, y + h * 0.26, subtitle, ha="center", va="center",
                fontsize=6.9, color=INK_MUTED, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=7.6, color=INK, zorder=4)


def connect(x0, y0, x1, y1, color, label, elbow=0.45, label_dy=1.9,
            label_dx=0.0):
    """Orthogonal elbow connector with the branch condition set beside it.

    Runs right from the parent, turns once, then enters the child horizontally
    so that every connector in the tree shares the same geometry.
    """
    xm = x0 + (x1 - x0) * elbow
    ax.plot([x0, xm], [y0, y0], color=color, linewidth=1.0, zorder=2,
            solid_capstyle="round")
    ax.plot([xm, xm], [y0, y1], color=color, linewidth=1.0, zorder=2,
            solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(
        (xm, y1), (x1, y1), arrowstyle="-|>", mutation_scale=7.5,
        linewidth=1.0, color=color, zorder=2, shrinkA=0.0, shrinkB=1.5))
    # label rides the horizontal entry segment, right-aligned so it runs away
    # from the child box instead of under it
    ax.text(x1 - 1.4 + label_dx, y1 + label_dy, label,
            ha="right", va="bottom" if label_dy >= 0 else "top",
            fontsize=6.7, color=INK_MUTED, zorder=5)


# stage 1 -------------------------------------------------------------------
node(0.0, 41, 25.0, 18, r"draw $\rho \sim U(0,1)$", "for this variable",
     NEUTRAL, "#f4f4f2", bold_title=False)

# exploit branch (top)
node(45, 70, 55, 19,
     "exploit — unchanged SPBO rule",
     r"$\rho \geq \epsilon$   ·   Equation (3)", NEUTRAL, "#f1f1ee")
connect(25.0, 50.0, 45, 79.5, NEUTRAL, r"$1-\epsilon = 0.9$", elbow=0.34)

# explore branch (bottom) -> stage 2 node
node(32, 11, 27, 19, "explore", r"draw $\rho_L \sim U(0,1)$",
     BLUE, "#eef4fc", bold_title=True)
connect(25.0, 50.0, 32, 20.5, BLUE, r"$\epsilon = 0.1$", elbow=0.36,
        label_dy=-2.1)

# stage 2 -------------------------------------------------------------------
node(69, 30, 31, 18, "Lévy jump", "Equation (8)", BLUE, "#eef4fc")
connect(59.0, 20.5, 69, 39.0, BLUE, r"$\rho_L \geq P_{\mathrm{th}}$",
        elbow=0.34, label_dy=1.7)

node(69, 2, 31, 18, "uniform spread", "Equation (9)", ORANGE, "#fdf0ea")
connect(59.0, 20.5, 69, 11.0, ORANGE, r"$\rho_L < P_{\mathrm{th}}$",
        elbow=0.34, label_dy=-2.1)



# ===========================================================================
# Panel (b): resulting marginal share of variable updates
# ===========================================================================
bx = fig.add_subplot(gs[0, 1])

labels = ["exploit\n(unchanged SPBO)", "uniform spread\nEq. (9)", "Lévy jump\nEq. (8)"]
values = [SHARE_EXPLOIT, SHARE_SPREAD, SHARE_LEVY]
colors = [NEUTRAL, ORANGE, BLUE]
ypos = [2, 1, 0]

bars = bx.barh(ypos, values, height=0.52, color=colors, zorder=3)
for bar in bars:                      # 4 px rounded data-end, flat at baseline
    bar.set_joinstyle("round")
    bar.set_capstyle("round")

for y, v in zip(ypos, values):
    bx.text(v + 2.0, y, f"{v:.0f}%", va="center", ha="left",
            fontsize=7.6, fontweight="bold", color=INK, zorder=4)

bx.set_yticks(ypos)
bx.set_yticklabels(labels, fontsize=7.0, color=INK)
bx.set_xlim(0, 100)
bx.set_ylim(-0.62, 2.62)
bx.set_xticks([0, 25, 50, 75, 100])
bx.tick_params(axis="x", labelsize=7.0, colors=INK_MUTED, length=0, pad=3)
bx.tick_params(axis="y", length=0, pad=4)
bx.set_xlabel("share of decision variables (%)", fontsize=7.2, color=INK_MUTED,
              labelpad=4)

bx.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
bx.set_axisbelow(True)
for side in ("top", "right", "left"):
    bx.spines[side].set_visible(False)
bx.spines["bottom"].set_color(GRID)
bx.spines["bottom"].set_linewidth(0.6)

# panel headers share one figure-space baseline so they always align
HDR_Y = 0.955
fig.text(0.008, HDR_Y, "(a)", fontsize=8.6, fontweight="bold", color=INK,
         va="center")
fig.text(0.037, HDR_Y, "Per-variable decision, average-student group",
         fontsize=8.0, color=INK, va="center")
fig.text(0.615, HDR_Y, "(b)", fontsize=8.6, fontweight="bold", color=INK,
         va="center")
fig.text(0.644, HDR_Y, "Resulting share of variable updates",
         fontsize=8.0, color=INK, va="center")
bx.text(0.985, 0.50, r"$\epsilon = 0.1$,   $P_{\mathrm{th}} = 0.6$",
        transform=bx.transAxes, fontsize=6.9, color=INK_MUTED,
        ha="right", va="center")

fig.patch.set_facecolor(SURFACE)
fig.savefig("fig_s4_epsilon_greedy.pdf", facecolor=SURFACE)
fig.savefig("fig_s4_epsilon_greedy.png", dpi=300, facecolor=SURFACE)
print("wrote fig_s4_epsilon_greedy.pdf / .png")
