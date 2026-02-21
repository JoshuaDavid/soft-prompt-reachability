"""
Experiment: High-Density Token Embedding Reachability

Tests whether restricting the token mixture optimizer to high-density
regions of the embedding manifold still achieves similar reachability.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from common import (
    model, tokenizer, DEVICE, D_MODEL, SEQ_LEN, TARGET_LAYER,
    RESULTS_DIR, get_residual, get_real_target, cosine_sim_per_position,
    load_real_sequences,
)

TAU = 0.5
STEPS = 2000
N_RESTARTS = 2
N_TARGETS = 25
KNN_K = 50
DENSITY_PERCENTILES = [10, 25, 50, 75, 100]

# ── Step 1: Compute token embedding density ─────────────────────────

print("=== Computing Token Embedding Density ===\n")

embed_matrix = model.gpt_neox.embed_in.weight.detach()  # [vocab, 768]
vocab_size = embed_matrix.shape[0]
print(f"Vocabulary size: {vocab_size}")

# Normalize for cosine similarity computation
embed_norm = F.normalize(embed_matrix, dim=-1)  # [vocab, 768]

# Compute pairwise cosine similarities in batches to avoid OOM
print("Computing k-NN density scores (batched)...")
density_scores = torch.zeros(vocab_size, device=DEVICE)
batch_size = 1024

for start in range(0, vocab_size, batch_size):
    end = min(start + batch_size, vocab_size)
    # [batch, vocab] cosine similarities
    sims = embed_norm[start:end] @ embed_norm.T
    # For each token, get top-k+1 (including self), take k neighbors
    topk_vals, _ = sims.topk(KNN_K + 1, dim=-1)
    # Exclude self (first column), average remaining k
    density_scores[start:end] = topk_vals[:, 1:].mean(dim=-1)
    if (end // batch_size) % 10 == 0:
        print(f"  Processed {end}/{vocab_size} tokens")

print(f"Density scores: min={density_scores.min():.4f}, max={density_scores.max():.4f}, "
      f"mean={density_scores.mean():.4f}")

# ── Step 2: Create density masks ────────────────────────────────────

print("\nCreating density masks...")
masks = {}
for pct in DENSITY_PERCENTILES:
    if pct == 100:
        masks[pct] = torch.ones(vocab_size, dtype=torch.bool, device=DEVICE)
    else:
        threshold = torch.quantile(density_scores, 1.0 - pct / 100.0)
        masks[pct] = density_scores >= threshold
    n_tokens = masks[pct].sum().item()
    # Show sample tokens from this density band
    allowed_ids = masks[pct].nonzero().squeeze(-1)[:10]
    sample_tokens = [tokenizer.decode([tid]) for tid in allowed_ids.tolist()]
    print(f"  Top {pct}%: {n_tokens} tokens (threshold={threshold if pct < 100 else 'none'}), "
          f"samples: {sample_tokens}")

# ── Step 3: Load targets ────────────────────────────────────────────

print(f"\nLoading {N_TARGETS} real targets...")
sequences = load_real_sequences(N_TARGETS)
targets = []
for seq in sequences[:N_TARGETS]:
    targets.append(get_real_target(seq))
print(f"Loaded {len(targets)} targets")

# ── Step 4: Token mixture optimization with density constraints ─────

def optimize_token_mixture(target, mask, tau=TAU, steps=STEPS, lr=0.05, seed=None):
    """Optimize softmax logits over vocabulary, masked to allowed tokens."""
    if seed is not None:
        torch.manual_seed(seed)

    logits = torch.randn(SEQ_LEN, vocab_size, device=DEVICE) * 0.01
    logits.requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=lr)

    # Precompute mask for -inf
    neg_inf_mask = ~mask  # tokens to exclude

    for step in range(steps):
        optimizer.zero_grad()

        # Mask out disallowed tokens
        masked_logits = logits.clone()
        masked_logits[:, neg_inf_mask] = -1e9

        # Soft token mixture
        weights = F.softmax(masked_logits / tau, dim=-1)  # [SEQ_LEN, vocab]
        soft_embeds = weights @ embed_matrix  # [SEQ_LEN, 768]

        residual = get_residual(soft_embeds)
        cos_loss = 1 - cosine_sim_per_position(residual, target).mean()
        mse_loss = ((residual - target) ** 2).mean()
        loss = cos_loss + 0.01 * mse_loss
        loss.backward()
        optimizer.step()

    # Evaluate continuous result
    with torch.no_grad():
        masked_logits = logits.clone()
        masked_logits[:, neg_inf_mask] = -1e9
        weights = F.softmax(masked_logits / tau, dim=-1)
        soft_embeds = weights @ embed_matrix
        continuous_residual = get_residual(soft_embeds)
        continuous_cos = cosine_sim_per_position(continuous_residual, target).mean().item()

        # Discrete: argmax tokens
        token_ids = masked_logits.argmax(dim=-1)
        discrete_embeds = embed_matrix[token_ids]
        discrete_residual = get_residual(discrete_embeds)
        discrete_cos = cosine_sim_per_position(discrete_residual, target).mean().item()

        # Entropy of distribution
        entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1).mean().item()

    return {
        "continuous_cos": continuous_cos,
        "discrete_cos": discrete_cos,
        "entropy": entropy,
        "token_ids": token_ids.cpu().tolist(),
    }


print("\n=== Running Density-Constrained Token Mixture Optimization ===\n")

all_results = {}
start_time = time.time()

for pct in DENSITY_PERCENTILES:
    mask = masks[pct]
    n_allowed = mask.sum().item()
    print(f"--- Top {pct}% density ({n_allowed} tokens) ---")

    pct_results = []
    for ti, target in enumerate(targets):
        best_result = None
        best_discrete = -1
        for restart in range(N_RESTARTS):
            result = optimize_token_mixture(target, mask, seed=restart * 1000 + ti)
            if result["discrete_cos"] > best_discrete:
                best_discrete = result["discrete_cos"]
                best_result = result
        pct_results.append(best_result)

        if (ti + 1) % 5 == 0:
            elapsed = time.time() - start_time
            median_d = np.median([r["discrete_cos"] for r in pct_results])
            print(f"  Target {ti+1}/{N_TARGETS}, median_discrete={median_d:.4f}, elapsed: {elapsed:.0f}s")

    continuous_vals = [r["continuous_cos"] for r in pct_results]
    discrete_vals = [r["discrete_cos"] for r in pct_results]

    all_results[pct] = {
        "n_tokens": n_allowed,
        "median_continuous_cos": float(np.median(continuous_vals)),
        "median_discrete_cos": float(np.median(discrete_vals)),
        "mean_discrete_cos": float(np.mean(discrete_vals)),
        "std_discrete_cos": float(np.std(discrete_vals)),
        "min_discrete_cos": float(np.min(discrete_vals)),
        "all_continuous": continuous_vals,
        "all_discrete": discrete_vals,
    }

    # Decode sample tokens from one result
    sample_ids = pct_results[0]["token_ids"][:10]
    sample_tokens = [tokenizer.decode([tid]) for tid in sample_ids]
    all_results[pct]["sample_tokens"] = sample_tokens

    print(f"  Results: continuous={np.median(continuous_vals):.4f}, "
          f"discrete={np.median(discrete_vals):.4f}")
    print(f"  Sample tokens: {sample_tokens}")
    print()

# ── Step 5: Save results ────────────────────────────────────────────

# Clean for JSON serialization
json_results = {}
for pct, data in all_results.items():
    json_results[str(pct)] = {k: v for k, v in data.items()}

with open(RESULTS_DIR / "density_constrained.json", "w") as f:
    json.dump(json_results, f, indent=2)
print("Saved results/density_constrained.json")

# ── Step 6: Plot ────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Discrete cosine vs density threshold
ax = axes[0]
pcts = sorted(all_results.keys())
medians = [all_results[p]["median_discrete_cos"] for p in pcts]
means = [all_results[p]["mean_discrete_cos"] for p in pcts]
stds = [all_results[p]["std_discrete_cos"] for p in pcts]
n_tokens = [all_results[p]["n_tokens"] for p in pcts]

ax.errorbar(pcts, means, yerr=stds, fmt='bo-', capsize=5, linewidth=2, markersize=8,
            label='Mean discrete cos')
ax.plot(pcts, medians, 'rs--', markersize=8, label='Median discrete cos')
ax.set_xlabel('Top-N% Density Tokens Allowed')
ax.set_ylabel('Discrete Cosine Similarity')
ax.set_title('Reachability vs Embedding Density Constraint')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(pcts)

# Plot 2: Box plot of all discrete cosines
ax = axes[1]
positions = list(range(len(pcts)))
bp = ax.boxplot([all_results[p]["all_discrete"] for p in pcts],
                positions=positions, widths=0.6, patch_artist=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(pcts)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticks(positions)
ax.set_xticklabels([f"Top {p}%\n({all_results[p]['n_tokens']})" for p in pcts], fontsize=9)
ax.set_ylabel('Discrete Cosine Similarity')
ax.set_title('Distribution of Discrete Cosine by Density Band')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Density score distribution with thresholds
ax = axes[2]
density_cpu = density_scores.cpu().numpy()
ax.hist(density_cpu, bins=100, alpha=0.7, edgecolor='black', linewidth=0.5)
for pct in [10, 25, 50, 75]:
    threshold = np.percentile(density_cpu, 100 - pct)
    ax.axvline(threshold, color='red', linestyle='--', alpha=0.7,
               label=f'Top {pct}% (>{threshold:.3f})')
ax.set_xlabel('k-NN Density Score (avg cos to 50 nearest neighbors)')
ax.set_ylabel('Count')
ax.set_title('Token Embedding Density Distribution')
ax.legend(fontsize=8)

fig.suptitle('Reachability Restricted to High-Density Embedding Regions', fontsize=14)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "density_constrained.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved results/density_constrained.png")

# ── Summary ─────────────────────────────────────────────────────────

print("\n=== Summary ===")
print(f"{'Density Band':>15} {'N Tokens':>10} {'Median Discrete':>16} {'Mean Discrete':>14}")
print("-" * 60)
for pct in pcts:
    d = all_results[pct]
    print(f"{'Top '+str(pct)+'%':>15} {d['n_tokens']:>10} {d['median_discrete_cos']:>16.4f} "
          f"{d['mean_discrete_cos']:>14.4f}")

elapsed = time.time() - start_time
print(f"\nTotal time: {elapsed:.0f}s")
