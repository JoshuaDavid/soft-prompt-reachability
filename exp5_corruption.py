"""
Experiment 5: Target Corruption Sweep (PCA Variant B)
Finds the transition boundary between reachable and unreachable.
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
    DEVICE, D_MODEL, SEQ_LEN, TARGET_LAYER, RESULTS_DIR,
    optimize_with_restarts, save_metrics,
)

N_TARGETS = 20
N_RESTARTS = 2
STEPS = 1000
FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def run_experiment5():
    print("=" * 70)
    print(f"EXPERIMENT 5: Target Corruption Sweep (PCA Variant)")
    print(f"  {N_TARGETS} targets, {len(FRACTIONS)} fractions, {N_RESTARTS} restarts, {STEPS} steps")
    print("=" * 70)

    exp_dir = RESULTS_DIR / "experiment5"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load targets and stats
    state = torch.load(RESULTS_DIR / "experiment1" / "state.pt", weights_only=False)
    targets = [t.to(DEVICE) for t in state["targets"][:N_TARGETS]]
    stats = torch.load(RESULTS_DIR / "activation_stats.pt", weights_only=False)

    # PCA of all residuals
    all_res = stats["all_residuals"]  # [N_seq, SEQ_LEN, D_MODEL]
    flat = all_res.reshape(-1, D_MODEL)
    flat_mean = flat.mean(dim=0)
    flat_centered = flat - flat_mean
    U, S, Vh = torch.linalg.svd(flat_centered, full_matrices=False)
    pca_dirs = Vh  # [D_MODEL, D_MODEL]

    # Empirical distribution of projections
    projections = flat_centered @ pca_dirs.T
    proj_mean = projections.mean(dim=0)
    proj_std = projections.std(dim=0)

    # Run sweeps
    all_frac_results = {f: [] for f in FRACTIONS}
    torch.manual_seed(777)
    t0 = time.time()
    total_opts = N_TARGETS * len(FRACTIONS)
    opt_count = 0

    for fi, frac in enumerate(FRACTIONS):
        n_dirs = int(frac * D_MODEL)
        print(f"\n--- Fraction {frac:.1f} ({n_dirs} PCA dirs replaced) ---")

        for ti, target in enumerate(targets):
            target_cpu = target.cpu()
            target_centered = target_cpu - flat_mean

            if n_dirs > 0:
                proj = target_centered @ pca_dirs[:n_dirs].T  # [SEQ_LEN, n_dirs]
                new_proj = torch.randn_like(proj) * proj_std[:n_dirs] + proj_mean[:n_dirs]
                diff = (new_proj - proj) @ pca_dirs[:n_dirs]
                corrupted = target_cpu + diff
            else:
                corrupted = target_cpu.clone()

            corrupted = corrupted.to(DEVICE)
            opt_count += 1
            r = optimize_with_restarts(corrupted, n_restarts=N_RESTARTS, steps=STEPS,
                                       log_every=0)
            all_frac_results[frac].append(r)

            if (ti + 1) % 10 == 0:
                elapsed = time.time() - t0
                remaining = elapsed / opt_count * (total_opts - opt_count)
                print(f"  Target {ti+1}/{N_TARGETS}, "
                      f"cos={r['best']['final_cosine']:.4f}, "
                      f"elapsed: {elapsed:.0f}s, ETA: {remaining:.0f}s")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    medians, q25s, q75s = [], [], []
    for frac in FRACTIONS:
        vals = [r["best"]["final_cosine"] for r in all_frac_results[frac]]
        medians.append(np.median(vals))
        q25s.append(np.percentile(vals, 25))
        q75s.append(np.percentile(vals, 75))

    ax1.plot(FRACTIONS, medians, 'bo-', linewidth=2, markersize=8, label='Median')
    ax1.fill_between(FRACTIONS, q25s, q75s, alpha=0.2, color='blue', label='IQR')
    ax1.set_xlabel('Fraction of PCA Directions Replaced')
    ax1.set_ylabel('Final Cosine Similarity')
    ax1.set_title('Reachability vs Target Corruption')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    box_data = [[r["best"]["final_cosine"] for r in all_frac_results[f]] for f in FRACTIONS]
    bp = ax2.boxplot(box_data, positions=range(len(FRACTIONS)), widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#2196F3')
        patch.set_alpha(0.6)
    ax2.set_xticklabels([f'{f:.1f}' for f in FRACTIONS])
    ax2.set_xlabel('Fraction Replaced')
    ax2.set_ylabel('Final Cosine Similarity')
    ax2.set_title('Distribution by Fraction')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 5: Target Corruption Sweep', fontsize=14)
    fig.tight_layout()
    fig.savefig(exp_dir / "corruption_curve.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save metrics
    metrics = {
        "fractions": FRACTIONS,
        "n_targets": N_TARGETS,
        "median_cosine_by_fraction": {str(f): float(m) for f, m in zip(FRACTIONS, medians)},
        "all_cosines_by_fraction": {
            str(f): [r["best"]["final_cosine"] for r in all_frac_results[f]]
            for f in FRACTIONS
        },
    }
    with open(exp_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nMedian cosine by fraction:")
    for f, m in zip(FRACTIONS, medians):
        print(f"  f={f:.1f}: {m:.4f}")

    return all_frac_results


if __name__ == "__main__":
    results = run_experiment5()
    print("\nExperiment 5 complete!")
