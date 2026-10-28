"""Build a plot + markdown summary from results.json.

Reads results.json and writes vad_sweep.png / controllability.png /
SUMMARY.md to the same directory by default (this script's own folder),
matching where steer.py wrote results.json. Override both with --out-dir
or ESTEER_OUT_DIR if steer.py was run with a different --out-dir (e.g. on
DSMLP)."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_OUT_DIR = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out-dir", default=os.environ.get("ESTEER_OUT_DIR", DEFAULT_OUT_DIR),
                help="directory with results.json, and where plots/SUMMARY.md are written")
args = p.parse_args()
OUT_DIR = args.out_dir

with open(os.path.join(OUT_DIR, "results.json")) as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, axis in zip(axes, ["valence", "arousal", "dominance"]):
    rows = data["logic_results"][axis]
    xs = [r["coeff"] for r in rows]
    ys = [r["accuracy"] * 100 for r in rows]
    ax.plot(xs, ys, marker="o")
    ax.set_title(axis.capitalize())
    ax.set_xlabel("steering coefficient")
    ax.axhline(ys[xs.index(0)], color="gray", linestyle="--", linewidth=0.8)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Logic task accuracy (%)")
fig.suptitle(f"E-STEER proof-of-concept: VAD steering vs. logic-reasoning accuracy\n({data['model']}, layer {data['layer']})")
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, "vad_sweep.png")
fig.savefig(fig_path, dpi=150)
print("saved", fig_path)

ctrl = data["controllability"]
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(ctrl["coeffs"], ctrl["prompt_scores"], marker="o", label=f"prompt-level (r={ctrl['r_prompt']:.2f})")
ax2.plot(ctrl["coeffs"], ctrl["steer_scores"], marker="s", label=f"activation steering (r={ctrl['r_steer']:.2f})")
ax2.set_xlabel("target valence coefficient")
ax2.set_ylabel("lexicon valence score of output")
ax2.set_title("Linear controllability: prompt vs. steering\n(analogous to paper's Table 1)")
ax2.legend()
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2_path = os.path.join(OUT_DIR, "controllability.png")
fig2.savefig(fig2_path, dpi=150)
print("saved", fig2_path)

# markdown summary
lines = ["# E-STEER proof-of-concept — results\n"]
lines.append(f"Model: `{data['model']}`, steering layer: {data['layer']}, "
              f"typical residual norm: {data['typical_norm']:.2f}\n")

lines.append("## Logic-reasoning accuracy under VAD steering\n")
lines.append("| axis | " + " | ".join(str(r['coeff']) for r in data["logic_results"]["valence"]) + " |")
lines.append("|---|" + "---|" * len(data["logic_results"]["valence"]))
for axis in ["valence", "arousal", "dominance"]:
    rows = data["logic_results"][axis]
    lines.append(f"| {axis} | " + " | ".join(f"{r['accuracy']*100:.0f}%" for r in rows) + " |")

lines.append("\n## Prompt-level vs. activation-steering linear controllability\n")
lines.append("| valence coeff | " + " | ".join(str(c) for c in ctrl["coeffs"]) + " | correlation (r) |")
lines.append("|---|" + "---|" * len(ctrl["coeffs"]) + "---|")
lines.append("| prompt-level | " + " | ".join(f"{v:+.2f}" for v in ctrl["prompt_scores"]) + f" | {ctrl['r_prompt']:.3f} |")
lines.append("| activation steering | " + " | ".join(f"{v:+.2f}" for v in ctrl["steer_scores"]) + f" | {ctrl['r_steer']:.3f} |")

with open(os.path.join(OUT_DIR, "SUMMARY.md"), "w") as f:
    f.write("\n".join(lines))
print("wrote SUMMARY.md")
