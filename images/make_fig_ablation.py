#!/usr/bin/env python3
"""Figure: eps-L-SPBO ablation sensitivity (real runs, ablation_parsed.csv)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
                     "font.size":8,"pdf.fonttype":42})
d = pd.read_csv("../ablation_parsed.csv")

def series(fam, pref):
    g = d[d.family==fam].groupby("config").agg(m=("accuracy","mean"), s=("accuracy","std"),
                                               nf=("n_features","mean"), nfs=("n_features","std"))
    g["x"] = [float(i.split("=")[1]) for i in g.index]
    return g.sort_values("x")

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
for ax, (fam, xlabel) in zip(axes, [("B", r"exploration probability $\epsilon$"),
                                    ("C", r"Lévy shape parameter $\beta$")]):
    g = series(fam, "")
    ax.errorbar(g.x, g.m, yerr=g.s, color=BLUE, marker="o", markersize=4.5, linewidth=2,
                capsize=3, elinewidth=1, zorder=3, label="accuracy (%)")
    ax2 = ax.twiny() if False else None
    ax.set_xlabel(xlabel.replace("\\u00e9","é"), fontsize=7.4, color=MUTED)
    ax.set_ylabel("held-out accuracy (%)", fontsize=7.4, color=MUTED)
    ax.set_ylim(98.6, 99.9)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for sp in ("left","bottom"): ax.spines[sp].set_color(GRID); ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(labelsize=7, colors=MUTED, length=0)

axes[0].text(0.0,1.10,"(a)",transform=axes[0].transAxes,fontsize=8.6,fontweight="bold",color=INK)
axes[0].text(0.055,1.10,"Sensitivity to $\\epsilon$",transform=axes[0].transAxes,fontsize=8,color=INK)
axes[1].text(0.0,1.10,"(b)",transform=axes[1].transAxes,fontsize=8.6,fontweight="bold",color=INK)
axes[1].text(0.055,1.10,"Sensitivity to $\\beta$",transform=axes[1].transAxes,fontsize=8,color=INK)
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig("fig_ablation_sensitivity.pdf"); fig.savefig("fig_ablation_sensitivity.png",dpi=300)
print("ok")
