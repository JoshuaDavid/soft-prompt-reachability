"""
Experiment 3: Interpolation and Extrapolation
Probes the geometry of the reachable set.
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

N_PAIRS = 10
N_RESTARTS = 3
STEPS = 1000
ALPHAS = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def run_experiment3():
    print("=" * 70)
    print(f"EXPERIMENT 3: Interpolation and Extrapolation")
    print(f"  {N_PAIRS} pairs, {len(ALPHAS)} alphas, {N_RESTARTS} restarts, {STEPS} steps")
    print("=" * 70)

    exp_dir = RESULTS_DIR / "experiment3"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load targets from Experiment 1
    state = torch.load(RESULTS_DIR / "experiment1" / "state.pt", weights_only=False)
    targets = [t.to(DEVICE) for t in state["targets"]]
    n_available = len(targets)

    # Select pairs deterministically
    np.random.seed(999)
    pair_indices = []
    while len(pair_indices) < N_PAIRS:
        a = np.random.randint(n_available)
        b = np.random.randint(n_available)
        if a != b:
            pair_indices.append((a, b))

    # Run optimizations
    all_pair_results = []
    t0 = time.time()
    total_opts = N_PAIRS * len(ALPHAS)
    opt_count = 0

    for pi, (idx_a, idx_b) in enumerate(pair_indices):
        A = targets[idx_a]
        B = targets[idx_b]
        pair_results = {}

        for alpha in ALPHAS:
            interp_target = alpha * A + (1 - alpha) * B
            opt_count += 1
            print(f"  Pair {pi+1}/{N_PAIRS}, α={alpha:+.2f} "
                  f"[{opt_count}/{total_opts}]:", end=" ")
            r = optimize_with_restarts(interp_target, n_restarts=N_RESTARTS, steps=STEPS)
            pair_results[alpha] = r
            print(f"cos={r['best']['final_cosine']:.4f}")

        all_pair_results.append(pair_results)
        elapsed = time.time() - t0
        remaining = elapsed / (pi + 1) * (N_PAIRS - pi - 1)
        print(f"  Pair {pi+1} done. Elapsed: {elapsed:.0f}s, ETA: {remaining:.0f}s\n")

    # Plot reachability curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Individual curves
    for pi, pair_results in enumerate(all_pair_results):
        cos_by_alpha = [pair_results[a]["best"]["final_cosine"] for a in ALPHAS]
        ax1.plot(ALPHAS, cos_by_alpha, 'o-', alpha=0.25, color='steelblue', markersize=3)

    # Median curve
    median_by_alpha = []
    q25_by_alpha = []
    q75_by_alpha = []
    for a in ALPHAS:
        vals = [pr[a]["best"]["final_cosine"] for pr in all_pair_results]
        median_by_alpha.append(np.median(vals))
        q25_by_alpha.append(np.percentile(vals, 25))
        q75_by_alpha.append(np.percentile(vals, 75))

    ax1.plot(ALPHAS, median_by_alpha, 'ro-', linewidth=2, markersize=8, label='Median', zorder=5)
    ax1.fill_between(ALPHAS, q25_by_alpha, q75_by_alpha, alpha=0.15, color='red', label='IQR')
    ax1.axvspan(0, 1, alpha=0.08, color='green', label='Interpolation range')
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Final Cosine Similarity')
    ax1.set_title('Reachability vs Interpolation Parameter')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot view
    box_data = [[pr[a]["best"]["final_cosine"] for pr in all_pair_results] for a in ALPHAS]
    bp = ax2.boxplot(box_data, positions=range(len(ALPHAS)), widths=0.6, patch_artist=True)
    for i, (patch, alpha) in enumerate(zip(bp['boxes'], ALPHAS)):
        if 0 <= alpha <= 1:
            patch.set_facecolor('#4CAF50')
        else:
            patch.set_facecolor('#FF9800')
        patch.set_alpha(0.6)
    ax2.set_xticklabels([f'{a:.2f}' for a in ALPHAS])
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('Final Cosine Similarity')
    ax2.set_title('Distribution by Alpha')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 3: Interpolation/Extrapolation Reachability', fontsize=14)
    fig.tight_layout()
    fig.savefig(exp_dir / "reachability_curve.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # MSE by alpha
    fig, ax = plt.subplots(figsize=(10, 6))
    median_mse_by_alpha = []
    for a in ALPHAS:
        vals = [pr[a]["best"]["final_mse"] for pr in all_pair_results]
        median_mse_by_alpha.append(np.median(vals))
    ax.plot(ALPHAS, median_mse_by_alpha, 'bs-', linewidth=2, markersize=8)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Median Final MSE')
    ax.set_title('MSE vs Interpolation Parameter')
    ax.axvspan(0, 1, alpha=0.08, color='green', label='Interpolation range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(exp_dir / "mse_by_alpha.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save metrics
    metrics = {
        "alphas": ALPHAS,
        "n_pairs": N_PAIRS,
        "pair_indices": [(int(a), int(b)) for a, b in pair_indices],
        "median_cosine_by_alpha": {str(a): float(v) for a, v in zip(ALPHAS, median_by_alpha)},
        "median_mse_by_alpha": {str(a): float(v) for a, v in zip(ALPHAS, median_mse_by_alpha)},
        "all_cosines_by_alpha": {
            str(a): [pr[a]["best"]["final_cosine"] for pr in all_pair_results]
            for a in ALPHAS
        },
    }
    with open(exp_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nMedian cosine by alpha:")
    for a, m in zip(ALPHAS, median_by_alpha):
        marker = " <-- boundary" if a in [-0.5, 1.5, 2.0] else ""
        print(f"  α={a:+.2f}: {m:.4f}{marker}")

    return all_pair_results


if __name__ == "__main__":
    results = run_experiment3()
    print("\nExperiment 3 complete!")
