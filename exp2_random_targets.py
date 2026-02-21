"""
Experiment 2a: Distribution-Matched Random Targets
Experiment 2b: Raw Random Targets
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
    get_residual, get_real_target, cosine_sim_per_position,
    optimize_with_restarts, load_real_sequences,
    save_metrics, plot_loss_curves, plot_cosine_histogram,
)

N_TARGETS = 50
N_RESTARTS = 3
STEPS_2A = 2000
STEPS_2B = 3000
N_STAT_SEQUENCES = 1000


def compute_activation_statistics():
    """Compute per-position mean and covariance of layer-6 residuals."""
    print(f"Computing activation statistics from {N_STAT_SEQUENCES} sequences...")
    sequences = load_real_sequences(N_STAT_SEQUENCES)
    print(f"  Loaded {len(sequences)} valid sequences")

    all_residuals = []
    for i, seq in enumerate(sequences):
        target = get_real_target(seq)
        all_residuals.append(target.cpu())
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sequences)} processed")

    stacked = torch.stack(all_residuals)  # [N, SEQ_LEN, D_MODEL]
    per_pos_mean = stacked.mean(dim=0)
    per_pos_norms = stacked.norm(dim=-1)
    typical_norms = per_pos_norms.mean(dim=0)

    # Per-position covariance
    per_pos_cov = []
    for pos in range(SEQ_LEN):
        data = stacked[:, pos, :]
        cov = torch.cov(data.T)
        per_pos_cov.append(cov)

    print(f"  Typical norms per position: {typical_norms.numpy().round(2)}")
    return {
        "mean": per_pos_mean,
        "covariances": per_pos_cov,
        "typical_norms": typical_norms,
        "all_residuals": stacked,
    }


def run_experiment2a(stats):
    """Distribution-matched random targets."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2a: Distribution-Matched Random Targets")
    print(f"  {N_TARGETS} targets, {N_RESTARTS} restarts, {STEPS_2A} steps")
    print("=" * 70)

    exp_dir = RESULTS_DIR / "experiment2a"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Generate targets from per-position Gaussian
    torch.manual_seed(42)
    targets = []
    for _ in range(N_TARGETS):
        positions = []
        for pos in range(SEQ_LEN):
            mu = stats["mean"][pos]
            cov = stats["covariances"][pos]
            try:
                L = torch.linalg.cholesky(cov + 1e-5 * torch.eye(D_MODEL))
                z = torch.randn(D_MODEL)
                sample = mu + L @ z
            except Exception:
                var = cov.diag().clamp(min=1e-6)
                sample = mu + torch.randn(D_MODEL) * var.sqrt()
            positions.append(sample)
        targets.append(torch.stack(positions).to(DEVICE))

    # Optimize
    all_results = []
    t0 = time.time()
    for i, target in enumerate(targets):
        print(f"\n2a Target {i+1}/{N_TARGETS}:")
        r = optimize_with_restarts(target, n_restarts=N_RESTARTS, steps=STEPS_2A,
                                   log_every=1000)
        all_results.append(r)
        elapsed = time.time() - t0
        remaining = elapsed / (i + 1) * (N_TARGETS - i - 1)
        print(f"  cos={r['best']['final_cosine']:.4f}, ETA: {remaining:.0f}s")

        if (i + 1) % 10 == 0:
            save_metrics(all_results, exp_dir / "metrics_checkpoint.json")

    # Outputs
    print("\n--- Generating Exp 2a plots ---")
    plot_loss_curves(all_results, exp_dir / "loss_curves.png",
                     "Exp 2a: Loss Curves (Distribution-Matched Random)")
    plot_cosine_histogram(all_results, exp_dir / "final_cosine_histogram.png",
                          "Exp 2a: Final Cosine Similarity")
    metrics = save_metrics(all_results, exp_dir / "metrics.json")
    print(f"Exp 2a Summary: {json.dumps(metrics['summary'], indent=2)}")

    return all_results


def run_experiment2b(stats):
    """Raw random targets (norm-matched)."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2b: Raw Random Targets")
    print(f"  {N_TARGETS} targets, {N_RESTARTS} restarts, {STEPS_2B} steps")
    print("=" * 70)

    exp_dir = RESULTS_DIR / "experiment2b"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Generate targets
    torch.manual_seed(123)
    targets = []
    for _ in range(N_TARGETS):
        raw = torch.randn(SEQ_LEN, D_MODEL)
        for pos in range(SEQ_LEN):
            raw[pos] = raw[pos] / raw[pos].norm() * stats["typical_norms"][pos]
        targets.append(raw.to(DEVICE))

    # Optimize
    all_results = []
    t0 = time.time()
    for i, target in enumerate(targets):
        print(f"\n2b Target {i+1}/{N_TARGETS}:")
        r = optimize_with_restarts(target, n_restarts=N_RESTARTS, steps=STEPS_2B,
                                   log_every=1000)
        all_results.append(r)
        elapsed = time.time() - t0
        remaining = elapsed / (i + 1) * (N_TARGETS - i - 1)
        print(f"  cos={r['best']['final_cosine']:.4f}, ETA: {remaining:.0f}s")

        if (i + 1) % 10 == 0:
            save_metrics(all_results, exp_dir / "metrics_checkpoint.json")

    # Outputs
    print("\n--- Generating Exp 2b plots ---")
    plot_loss_curves(all_results, exp_dir / "loss_curves.png",
                     "Exp 2b: Loss Curves (Raw Random)")
    plot_cosine_histogram(all_results, exp_dir / "final_cosine_histogram.png",
                          "Exp 2b: Final Cosine Similarity")
    metrics = save_metrics(all_results, exp_dir / "metrics.json")
    print(f"Exp 2b Summary: {json.dumps(metrics['summary'], indent=2)}")

    # Check if still converging
    cos_at_80pct = np.median([r["best"]["cosine_history"][int(STEPS_2B * 0.8)]
                              for r in all_results])
    cos_at_end = np.median([r["best"]["final_cosine"] for r in all_results])
    print(f"Convergence check: cos at 80% = {cos_at_80pct:.4f}, "
          f"at end = {cos_at_end:.4f}, delta = {cos_at_end - cos_at_80pct:.4f}")

    return all_results


if __name__ == "__main__":
    stats = compute_activation_statistics()
    # Save stats for potential use by Exp 5
    torch.save(stats, RESULTS_DIR / "activation_stats.pt")
    print(f"Saved activation statistics")

    results_2a = run_experiment2a(stats)
    results_2b = run_experiment2b(stats)
    print("\nExperiments 2a and 2b complete!")
