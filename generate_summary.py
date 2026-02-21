"""
Generate the summary plot and final analysis combining all experiments.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")


def load_metrics(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def generate_summary_plot():
    """The money plot: all experiments compared."""
    print("Generating summary plot...")

    data = {}
    labels = []
    colors = []

    # Exp 1
    m = load_metrics(RESULTS_DIR / "experiment1" / "metrics.json")
    if m:
        data["Real\nTargets"] = m["final_cosines"]
        labels.append("Real\nTargets")
        colors.append('#2196F3')

    # Exp 4
    m_noffn = load_metrics(RESULTS_DIR / "experiment4" / "metrics_no_ffn.json")
    m_noattn = load_metrics(RESULTS_DIR / "experiment4" / "metrics_no_attn.json")
    if m_noffn:
        data["No FFN"] = m_noffn["final_cosines"]
        labels.append("No FFN")
        colors.append('#F44336')
    if m_noattn:
        data["No Attn"] = m_noattn["final_cosines"]
        labels.append("No Attn")
        colors.append('#FF5722')

    # Exp 2a
    m = load_metrics(RESULTS_DIR / "experiment2a" / "metrics.json")
    if m:
        data["Dist-Matched\nRandom"] = m["final_cosines"]
        labels.append("Dist-Matched\nRandom")
        colors.append('#4CAF50')

    # Exp 2b
    m = load_metrics(RESULTS_DIR / "experiment2b" / "metrics.json")
    if m:
        data["Raw\nRandom"] = m["final_cosines"]
        labels.append("Raw\nRandom")
        colors.append('#FF9800')

    # Exp 3: selected alphas
    m3 = load_metrics(RESULTS_DIR / "experiment3" / "metrics.json")
    if m3:
        for alpha in ["-0.5", "0.5", "1.5"]:
            key = f"Interp\nα={alpha}"
            if alpha in m3.get("all_cosines_by_alpha", {}):
                data[key] = m3["all_cosines_by_alpha"][alpha]
                labels.append(key)
                colors.append('#9C27B0')

    if not data:
        print("No results to plot!")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.5), 7))
    bp = ax.boxplot([data[l] for l in labels], positions=range(len(labels)),
                    widths=0.6, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Final Cosine Similarity', fontsize=12)
    ax.set_title('Summary: Soft Prompt Reachability Across All Experiments', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add median annotations
    for i, label in enumerate(labels):
        med = np.median(data[label])
        ax.annotate(f'{med:.3f}', xy=(i, med), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=8,
                    fontweight='bold')

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'summary.png'}")


def print_final_analysis():
    """Print a structured analysis of all results."""
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    # Exp 1
    m = load_metrics(RESULTS_DIR / "experiment1" / "metrics.json")
    if m:
        s = m["summary"]
        print(f"\n[Exp 1] Real Targets:")
        print(f"  Median cosine: {s['median_cosine']:.4f}")
        print(f"  Mean cosine:   {s['mean_cosine']:.4f} ± {s['std_cosine']:.4f}")
        print(f"  Range: [{s['min_cosine']:.4f}, {s['max_cosine']:.4f}]")

    # Exp 1b
    m1b = load_metrics(RESULTS_DIR / "experiment1b" / "metrics.json")
    if m1b:
        c = m1b["cosine_to_nearest"]
        print(f"\n[Exp 1b] Embedding Manifold Distance:")
        print(f"  Median cos to nearest token: {c['median']:.4f}")
        print(f"  Frac with cos > 0.9: {c['frac_above_0.9']:.1%}")
        print(f"  Frac with cos > 0.95: {c['frac_above_0.95']:.1%}")
        if c['median'] > 0.9:
            print(f"  → Optimized vectors are CLOSE to real embeddings")
        else:
            print(f"  → Optimized vectors are FAR from real embeddings")

    # Exp 4
    m4 = load_metrics(RESULTS_DIR / "experiment4" / "ablation_summary.json")
    if m4:
        print(f"\n[Exp 4] Ablation Study:")
        print(f"  Full model median:  {m4['full_model']['median']:.4f}")
        print(f"  No-FFN median:      {m4['no_ffn']['median']:.4f} "
              f"(Δ={m4['no_ffn']['median_delta']:.4f})")
        print(f"  No-Attn median:     {m4['no_attn']['median']:.4f} "
              f"(Δ={m4['no_attn']['median_delta']:.4f})")
        nf = m4['no_ffn']
        na = m4['no_attn']
        if nf['median_delta'] < -0.01:
            print(f"  → FFN nonlinearities are NET-POSITIVE for reachability")
        elif nf['median_delta'] > 0.01:
            print(f"  → FFN nonlinearities are NET-NEGATIVE for reachability")
        else:
            print(f"  → FFN nonlinearities have minimal effect on reachability")
        if na['median_delta'] < -0.01:
            print(f"  → Attention coupling is NET-POSITIVE for reachability")
        elif na['median_delta'] > 0.01:
            print(f"  → Attention coupling is NET-NEGATIVE for reachability")
        else:
            print(f"  → Attention coupling has minimal effect on reachability")

    # Exp 2a/2b
    m2a = load_metrics(RESULTS_DIR / "experiment2a" / "metrics.json")
    m2b = load_metrics(RESULTS_DIR / "experiment2b" / "metrics.json")
    if m2a:
        print(f"\n[Exp 2a] Distribution-Matched Random:")
        print(f"  Median cosine: {m2a['summary']['median_cosine']:.4f}")
    if m2b:
        print(f"\n[Exp 2b] Raw Random:")
        print(f"  Median cosine: {m2b['summary']['median_cosine']:.4f}")
    if m2a and m2b and m:
        gap = m2a['summary']['median_cosine'] - m2b['summary']['median_cosine']
        print(f"\n  Gap (2a - 2b): {gap:.4f}")
        if gap > 0.05:
            print(f"  → Reachable set is CONCENTRATED near natural manifold")
        else:
            print(f"  → Reachable set extends substantially beyond natural manifold")

    # Exp 3
    m3 = load_metrics(RESULTS_DIR / "experiment3" / "metrics.json")
    if m3:
        mc = m3["median_cosine_by_alpha"]
        print(f"\n[Exp 3] Interpolation/Extrapolation:")
        for a in sorted(mc.keys(), key=float):
            print(f"  α={float(a):+.2f}: median cos = {mc[a]:.4f}")
        interp_vals = [mc[a] for a in ["0.25", "0.5", "0.75"] if a in mc]
        extrap_vals = [mc[a] for a in ["-0.5", "1.5", "2.0"] if a in mc]
        if interp_vals and extrap_vals:
            if np.mean(interp_vals) - np.mean(extrap_vals) > 0.05:
                print(f"  → Reachable set is approximately CONVEX near data manifold")
            else:
                print(f"  → Reachable set extends beyond convex hull")

    # Exp 5
    m5 = load_metrics(RESULTS_DIR / "experiment5" / "metrics.json")
    if m5:
        mc = m5["median_cosine_by_fraction"]
        print(f"\n[Exp 5] Corruption Sweep:")
        for f in sorted(mc.keys(), key=float):
            print(f"  f={float(f):.1f}: median cos = {mc[f]:.4f}")
        vals = [mc[str(f)] for f in [0.0, 0.5, 1.0] if str(f) in mc]
        if len(vals) >= 3:
            drop = vals[0] - vals[2]
            if drop > 0.1:
                print(f"  → Sharp degradation (drop={drop:.4f}): clear reachability boundary")
            else:
                print(f"  → Gradual degradation (drop={drop:.4f}): reachable set is 'thick'")


if __name__ == "__main__":
    generate_summary_plot()
    print_final_analysis()
