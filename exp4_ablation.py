"""
Experiment 4: FFN and Attention Ablation
Tests whether FFN nonlinearities or attention coupling are net-positive for reachability.
"""

import torch
import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from common import (
    model, DEVICE, D_MODEL, SEQ_LEN, TARGET_LAYER, RESULTS_DIR,
    get_residual, cosine_sim_per_position,
    optimize_with_restarts, install_ablation_hooks, remove_hooks,
    save_metrics, plot_loss_curves, plot_cosine_histogram,
)

N_RESTARTS = 3
STEPS = 1000


def run_experiment4():
    print("=" * 70)
    print("EXPERIMENT 4: FFN and Attention Ablation")
    print("=" * 70)

    exp_dir = RESULTS_DIR / "experiment4"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load targets from Experiment 1
    state = torch.load(RESULTS_DIR / "experiment1" / "state.pt", weights_only=False)
    targets = [t.to(DEVICE) for t in state["targets"]]
    exp1_results = state["results"]
    n_targets = len(targets)
    print(f"Loaded {n_targets} targets from Experiment 1")

    results = {"full": exp1_results}

    # ── 4a: FFN Ablated ──
    print(f"\n--- 4a: FFN Ablated (Attention Only), {n_targets} targets ---")
    hooks = install_ablation_hooks(ablate="mlp")
    results["no_ffn"] = []
    t0 = time.time()
    for i, target in enumerate(targets):
        print(f"  No-FFN target {i+1}/{n_targets}:", end=" ")
        r = optimize_with_restarts(target, n_restarts=N_RESTARTS, steps=STEPS)
        results["no_ffn"].append(r)
        elapsed = time.time() - t0
        remaining = elapsed / (i + 1) * (n_targets - i - 1)
        print(f"cos={r['best']['final_cosine']:.4f}, ETA: {remaining:.0f}s")
    remove_hooks(hooks)

    # ── 4b: Attention Ablated ──
    print(f"\n--- 4b: Attention Ablated (FFN Only), {n_targets} targets ---")
    hooks = install_ablation_hooks(ablate="attention")
    results["no_attn"] = []
    t0 = time.time()
    for i, target in enumerate(targets):
        print(f"  No-Attn target {i+1}/{n_targets}:", end=" ")
        r = optimize_with_restarts(target, n_restarts=N_RESTARTS, steps=STEPS)
        results["no_attn"].append(r)
        elapsed = time.time() - t0
        remaining = elapsed / (i + 1) * (n_targets - i - 1)
        print(f"cos={r['best']['final_cosine']:.4f}, ETA: {remaining:.0f}s")
    remove_hooks(hooks)

    # ── Generate outputs ──
    print("\n--- Generating Experiment 4 plots ---")

    full_cos = [r["best"]["final_cosine"] for r in results["full"]]
    noffn_cos = [r["best"]["final_cosine"] for r in results["no_ffn"]]
    noattn_cos = [r["best"]["final_cosine"] for r in results["no_attn"]]

    # Save metrics for each variant
    for variant in ["no_ffn", "no_attn"]:
        save_metrics(results[variant], exp_dir / f"metrics_{variant}.json")

    # Scatter plot: paired comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(full_cos, noffn_cos, alpha=0.6, s=30, c='green')
    lims = [min(min(full_cos), min(noffn_cos)) - 0.02, 1.01]
    ax1.plot(lims, lims, 'k--', alpha=0.3)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel('Full Model Cosine Sim')
    ax1.set_ylabel('No-FFN (Attn Only) Cosine Sim')
    ax1.set_title(f'4a: FFN Ablated\n(median Δ={np.median(np.array(noffn_cos) - np.array(full_cos)):.4f})')
    ax1.set_aspect('equal')

    ax2.scatter(full_cos, noattn_cos, alpha=0.6, s=30, c='orange')
    lims2 = [min(min(full_cos), min(noattn_cos)) - 0.02, 1.01]
    ax2.plot(lims2, lims2, 'k--', alpha=0.3)
    ax2.set_xlim(lims2)
    ax2.set_ylim(lims2)
    ax2.set_xlabel('Full Model Cosine Sim')
    ax2.set_ylabel('No-Attn (FFN Only) Cosine Sim')
    ax2.set_title(f'4b: Attention Ablated\n(median Δ={np.median(np.array(noattn_cos) - np.array(full_cos)):.4f})')
    ax2.set_aspect('equal')

    fig.suptitle('Experiment 4: Ablation Comparison', fontsize=14)
    fig.tight_layout()
    fig.savefig(exp_dir / "ablation_comparison.png", dpi=150, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / "ablation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Overlaid histograms
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(full_cos, bins=25, alpha=0.5, label=f'Full (med={np.median(full_cos):.4f})',
            color='blue', edgecolor='black')
    ax.hist(noffn_cos, bins=25, alpha=0.5, label=f'No-FFN (med={np.median(noffn_cos):.4f})',
            color='green', edgecolor='black')
    ax.hist(noattn_cos, bins=25, alpha=0.5, label=f'No-Attn (med={np.median(noattn_cos):.4f})',
            color='orange', edgecolor='black')
    ax.set_xlabel('Final Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Experiment 4: Ablation Comparison Histograms')
    ax.legend()
    fig.tight_layout()
    fig.savefig(exp_dir / "ablation_histograms.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Per-position analysis for each variant
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (variant, label, color) in zip(axes, [
        ("full", "Full Model", "blue"),
        ("no_ffn", "No FFN", "green"),
        ("no_attn", "No Attention", "orange"),
    ]):
        per_pos_all = []
        for r, t in zip(results[variant], targets):
            prompt = r["best"]["prompt"].to(DEVICE)
            with torch.no_grad():
                if variant == "no_ffn":
                    hooks = install_ablation_hooks(ablate="mlp")
                    achieved = get_residual(prompt)
                    remove_hooks(hooks)
                elif variant == "no_attn":
                    hooks = install_ablation_hooks(ablate="attention")
                    achieved = get_residual(prompt)
                    remove_hooks(hooks)
                else:
                    achieved = get_residual(prompt)
                cos = cosine_sim_per_position(achieved, t).cpu().numpy()
            per_pos_all.append(cos)
        per_pos_arr = np.array(per_pos_all)
        means = per_pos_arr.mean(axis=0)
        stds = per_pos_arr.std(axis=0)
        ax.errorbar(range(SEQ_LEN), means, yerr=stds, fmt='o-', capsize=3, color=color)
        ax.set_xlabel('Position')
        ax.set_title(f'{label}\n(mean={per_pos_arr.mean():.4f})')
        ax.set_xticks(range(SEQ_LEN))
    axes[0].set_ylabel('Cosine Similarity')
    fig.suptitle('Experiment 4: Per-Position Cosine by Ablation', fontsize=14)
    fig.tight_layout()
    fig.savefig(exp_dir / "per_position_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Delta analysis
    delta_noffn = np.array(noffn_cos) - np.array(full_cos)
    delta_noattn = np.array(noattn_cos) - np.array(full_cos)

    summary = {
        "full_model": {
            "mean": float(np.mean(full_cos)),
            "median": float(np.median(full_cos)),
            "std": float(np.std(full_cos)),
        },
        "no_ffn": {
            "mean": float(np.mean(noffn_cos)),
            "median": float(np.median(noffn_cos)),
            "std": float(np.std(noffn_cos)),
            "mean_delta": float(delta_noffn.mean()),
            "median_delta": float(np.median(delta_noffn)),
            "frac_better_than_full": float((delta_noffn > 0).mean()),
            "frac_worse_than_full": float((delta_noffn < 0).mean()),
        },
        "no_attn": {
            "mean": float(np.mean(noattn_cos)),
            "median": float(np.median(noattn_cos)),
            "std": float(np.std(noattn_cos)),
            "mean_delta": float(delta_noattn.mean()),
            "median_delta": float(np.median(delta_noattn)),
            "frac_better_than_full": float((delta_noattn > 0).mean()),
            "frac_worse_than_full": float((delta_noattn < 0).mean()),
        },
    }
    with open(exp_dir / "ablation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAblation Summary:")
    print(f"  Full model:  median cos = {np.median(full_cos):.4f}")
    print(f"  No FFN:      median cos = {np.median(noffn_cos):.4f} "
          f"(Δ = {np.median(delta_noffn):.4f})")
    print(f"  No Attention: median cos = {np.median(noattn_cos):.4f} "
          f"(Δ = {np.median(delta_noattn):.4f})")
    print(f"  No-FFN better than full: {(delta_noffn > 0).mean():.0%}")
    print(f"  No-Attn better than full: {(delta_noattn > 0).mean():.0%}")

    return results


if __name__ == "__main__":
    results = run_experiment4()
    print("\nExperiment 4 complete!")
