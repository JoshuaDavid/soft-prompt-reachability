"""
Experiment 1: Reachability of Real Activations
Experiment 1b: Embedding Manifold Distance
"""

import torch
import numpy as np
import json
import time
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from common import (
    model, tokenizer, DEVICE, D_MODEL, SEQ_LEN, TARGET_LAYER, NUM_LAYERS,
    RESULTS_DIR, get_residual, get_real_target, get_real_all_residuals,
    cosine_sim_per_position, optimize_with_restarts, load_real_sequences,
    save_metrics, plot_loss_curves, plot_cosine_histogram,
    plot_cosine_by_position, plot_layer_trajectories,
)

N_TARGETS = 50
N_RESTARTS = 3
STEPS = 1000

def run_experiment1():
    print("=" * 70)
    print(f"EXPERIMENT 1: Reachability of Real Activations")
    print(f"  {N_TARGETS} targets, {N_RESTARTS} restarts, {STEPS} steps")
    print("=" * 70)

    exp_dir = RESULTS_DIR / "experiment1"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading sequences...")
    sequences = load_real_sequences(N_TARGETS)
    print(f"Loaded {len(sequences)} sequences")

    # Compute targets
    print("Computing targets...")
    targets = []
    real_all_residuals_list = []
    for i, seq in enumerate(sequences):
        targets.append(get_real_target(seq))
        real_all_residuals_list.append([r.cpu() for r in get_real_all_residuals(seq)])

    # Run optimization
    all_results = []
    t0 = time.time()
    for i, target in enumerate(targets):
        print(f"\nTarget {i+1}/{N_TARGETS}:")
        result = optimize_with_restarts(target, n_restarts=N_RESTARTS, steps=STEPS,
                                        log_every=500)
        all_results.append(result)
        elapsed = time.time() - t0
        remaining = elapsed / (i + 1) * (N_TARGETS - i - 1)
        print(f"  Best cos: {result['best']['final_cosine']:.4f}, "
              f"var: {result['restart_variance']:.6f}, "
              f"ETA: {remaining:.0f}s")

        if (i + 1) % 10 == 0:
            save_metrics(all_results, exp_dir / "metrics_checkpoint.json")

    # Generate all outputs
    print("\n--- Generating Experiment 1 plots ---")
    plot_loss_curves(all_results, exp_dir / "loss_curves.png",
                     "Exp 1: Loss Curves (Real Targets)")
    plot_cosine_histogram(all_results, exp_dir / "final_cosine_histogram.png",
                          "Exp 1: Final Cosine Similarity (Real Targets)")
    plot_cosine_by_position(all_results, targets, exp_dir / "cosine_by_position.png",
                            "Exp 1: Cosine Similarity by Position")
    plot_layer_trajectories(all_results, targets, real_all_residuals_list,
                            exp_dir / "layer_trajectories.png",
                            "Exp 1: Layer Trajectories")
    metrics = save_metrics(all_results, exp_dir / "metrics.json")
    print(f"\nExp 1 Summary: {json.dumps(metrics['summary'], indent=2)}")

    # Check convergence adequacy
    cos_at_80pct = np.median([r["best"]["cosine_history"][int(STEPS * 0.8)]
                              for r in all_results])
    cos_at_end = np.median([r["best"]["final_cosine"] for r in all_results])
    print(f"Median cos at 80%: {cos_at_80pct:.4f}, at end: {cos_at_end:.4f}, "
          f"delta: {cos_at_end - cos_at_80pct:.6f}")

    # Save state for other experiments
    state = {
        "targets": [t.cpu() for t in targets],
        "sequences": [s.cpu() for s in sequences],
        "real_all_residuals": real_all_residuals_list,
        "results": all_results,
    }
    torch.save(state, exp_dir / "state.pt")
    print(f"Saved state to {exp_dir / 'state.pt'}")

    return all_results, targets, real_all_residuals_list, sequences


def run_experiment1b(all_results):
    """Analyze distance from optimized inputs to token embedding manifold."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1b: Embedding Manifold Distance")
    print("=" * 70)

    exp_dir = RESULTS_DIR / "experiment1b"
    exp_dir.mkdir(parents=True, exist_ok=True)

    embed_matrix = model.gpt_neox.embed_in.weight.detach()
    embed_normed = embed_matrix / embed_matrix.norm(dim=-1, keepdim=True)

    all_cos_sims = []
    all_l2_dists = []
    all_nearest_tokens = []
    optimized_norms = []

    for r in all_results:
        prompt = r["best"]["prompt"].to(DEVICE)
        prompt_normed = prompt / prompt.norm(dim=-1, keepdim=True)
        cos_sims = prompt_normed @ embed_normed.T
        best_cos, best_idx = cos_sims.max(dim=-1)
        best_embeds = embed_matrix[best_idx]
        l2_dists = (prompt - best_embeds).norm(dim=-1)

        all_cos_sims.append(best_cos.cpu().numpy())
        all_l2_dists.append(l2_dists.cpu().numpy())
        all_nearest_tokens.append(best_idx.cpu().numpy())
        optimized_norms.append(prompt.norm(dim=-1).cpu().numpy())

    all_cos = np.concatenate(all_cos_sims)
    all_l2 = np.concatenate(all_l2_dists)
    all_norms = np.concatenate(optimized_norms)

    # Compare to embedding norms
    embed_norms = embed_matrix.norm(dim=-1).cpu().numpy()

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(all_cos, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Cosine Sim to Nearest Token Embedding')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Nearest-Neighbor Cosine Similarity')
    axes[0, 0].axvline(np.median(all_cos), color='red', linestyle='--',
                        label=f'Median: {np.median(all_cos):.4f}')
    axes[0, 0].legend()

    axes[0, 1].hist(all_l2, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('L2 Distance to Nearest Token Embedding')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('L2 Distance Distribution')
    axes[0, 1].axvline(np.median(all_l2), color='red', linestyle='--',
                        label=f'Median: {np.median(all_l2):.4f}')
    axes[0, 1].legend()

    axes[1, 0].hist(all_norms, bins=50, edgecolor='black', alpha=0.7, color='green',
                     label=f'Optimized (med={np.median(all_norms):.2f})')
    axes[1, 0].hist(embed_norms, bins=50, edgecolor='black', alpha=0.5, color='blue',
                     label=f'Embeddings (med={np.median(embed_norms):.2f})')
    axes[1, 0].set_xlabel('L2 Norm')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Norm Comparison')
    axes[1, 0].legend()

    # Per-position cosine to nearest
    per_pos_cos = np.array(all_cos_sims)  # [n_targets, SEQ_LEN]
    means = per_pos_cos.mean(axis=0)
    stds = per_pos_cos.std(axis=0)
    axes[1, 1].errorbar(range(SEQ_LEN), means, yerr=stds, fmt='o-', capsize=3)
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('Cosine Sim to Nearest Embedding')
    axes[1, 1].set_title('Embedding Proximity by Position')
    axes[1, 1].set_xticks(range(SEQ_LEN))

    fig.suptitle('Experiment 1b: Embedding Manifold Distance Analysis', fontsize=14)
    fig.tight_layout()
    fig.savefig(exp_dir / "embedding_distance.png", dpi=150, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / "experiment1b_embedding_distance.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # PCA analysis
    embed_centered = embed_matrix - embed_matrix.mean(dim=0)
    U, S, Vh = torch.linalg.svd(embed_centered, full_matrices=False)
    explained_var = (S ** 2) / (S ** 2).sum()
    cumvar = explained_var.cumsum(0).cpu().numpy()

    pca_analysis = {}
    for k in [10, 50, 100, 200, 384]:
        basis = Vh[:k]
        recon_errors = []
        for r in all_results:
            prompt = r["best"]["prompt"].to(DEVICE)
            pc = prompt - embed_matrix.mean(dim=0)
            proj = pc @ basis.T @ basis
            err = (pc - proj).norm(dim=-1).cpu().numpy()
            recon_errors.append(err)
        recon_errors = np.concatenate(recon_errors)
        pca_analysis[f"k={k}"] = {
            "mean_recon_error": float(recon_errors.mean()),
            "median_recon_error": float(np.median(recon_errors)),
            "cumvar_explained": float(cumvar[k - 1]),
        }

    metrics = {
        "cosine_to_nearest": {
            "mean": float(all_cos.mean()),
            "median": float(np.median(all_cos)),
            "std": float(all_cos.std()),
            "min": float(all_cos.min()),
            "max": float(all_cos.max()),
            "frac_above_0.9": float((all_cos > 0.9).mean()),
            "frac_above_0.95": float((all_cos > 0.95).mean()),
        },
        "l2_to_nearest": {
            "mean": float(all_l2.mean()),
            "median": float(np.median(all_l2)),
            "std": float(all_l2.std()),
        },
        "optimized_norms": {
            "mean": float(all_norms.mean()),
            "median": float(np.median(all_norms)),
        },
        "embedding_norms": {
            "mean": float(embed_norms.mean()),
            "median": float(np.median(embed_norms)),
        },
        "pca_analysis": pca_analysis,
    }
    with open(exp_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Cosine to nearest embedding: mean={all_cos.mean():.4f}, "
          f"median={np.median(all_cos):.4f}")
    print(f"Frac cos > 0.9: {(all_cos > 0.9).mean():.3f}, "
          f"> 0.95: {(all_cos > 0.95).mean():.3f}")
    print(f"Optimized norms: mean={all_norms.mean():.2f} vs "
          f"embedding norms: mean={embed_norms.mean():.2f}")

    # Decode nearest tokens for a sample
    print("\nSample nearest tokens (first optimized prompt):")
    tokens = all_nearest_tokens[0]
    decoded = [tokenizer.decode([t]) for t in tokens]
    print(f"  {decoded}")

    return metrics


if __name__ == "__main__":
    results, targets, real_all_res, sequences = run_experiment1()
    exp1b_metrics = run_experiment1b(results)
    print("\nExperiment 1 + 1b complete!")
