"""
Experiment: Prefix Soft Prompt Steering

Can a 20-token soft prompt prefix steer a specific position's
mid-layer activation in a 100-token context?

Setup:
- Fixed 100-token real context (from WikiText-2)
- 20-token optimizable prefix prepended to context
- Target: match a specific layer-6 activation at position 120
  (i.e., position 100 within the context, 0-indexed from prefix start = position 119)

This tests whether soft prompting can reach arbitrary activations
at distant positions through the attention mechanism.
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
    model, tokenizer, DEVICE, D_MODEL, TARGET_LAYER, NUM_LAYERS,
    RESULTS_DIR, cosine_sim_per_position, load_real_sequences,
)

PREFIX_LEN = 20
CONTEXT_LEN = 100
TOTAL_LEN = PREFIX_LEN + CONTEXT_LEN
STEPS = 2000
N_RESTARTS = 3
LR = 0.01

# ── Helpers ─────────────────────────────────────────────────────────

def get_residual_fast_long(input_embeds, layer=TARGET_LAYER):
    """Partial forward for longer sequences."""
    hidden = input_embeds.unsqueeze(0)
    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
    position_embeddings = model.gpt_neox.rotary_emb(hidden, position_ids=position_ids)
    for i in range(layer):
        hidden = model.gpt_neox.layers[i](
            hidden, position_embeddings=position_embeddings,
        )
    return hidden.squeeze(0)


def get_all_residuals_long(input_embeds):
    """Full forward for longer sequences, return all layer residuals."""
    outputs = model(
        inputs_embeds=input_embeds.unsqueeze(0),
        output_hidden_states=True,
    )
    return [h.squeeze(0) for h in outputs.hidden_states]


def optimize_prefix(context_embeds, target_activation, target_positions,
                    steps=STEPS, lr=LR, seed=None, mse_weight=0.01):
    """Optimize a prefix to steer target_positions to target_activation.

    Args:
        context_embeds: [CONTEXT_LEN, D_MODEL] fixed context embeddings
        target_activation: [len(target_positions), D_MODEL] target at layer 6
        target_positions: list of position indices (in full sequence) to steer
        steps: optimization steps
        lr: learning rate
        seed: random seed
        mse_weight: MSE loss weight
    """
    if seed is not None:
        torch.manual_seed(seed)

    prefix = torch.randn(PREFIX_LEN, D_MODEL, device=DEVICE) * 0.1
    prefix.requires_grad_(True)
    optimizer = torch.optim.Adam([prefix], lr=lr)

    cosine_history = []
    for step in range(steps):
        optimizer.zero_grad()

        # Concatenate prefix + context
        full_embeds = torch.cat([prefix, context_embeds], dim=0)  # [120, 768]

        # Forward through layers 0..5
        residual = get_residual_fast_long(full_embeds)  # [120, 768]

        # Extract target positions
        achieved = residual[target_positions]  # [n_pos, 768]

        cos_loss = 1 - F.cosine_similarity(achieved, target_activation, dim=-1).mean()
        mse_loss = ((achieved - target_activation) ** 2).mean()
        loss = cos_loss + mse_weight * mse_loss
        loss.backward()
        optimizer.step()

        cosine_history.append(1 - cos_loss.item())

    with torch.no_grad():
        full_embeds = torch.cat([prefix.detach(), context_embeds], dim=0)
        residual = get_residual_fast_long(full_embeds)
        achieved = residual[target_positions]
        final_cos = F.cosine_similarity(achieved, target_activation, dim=-1).mean().item()
        per_pos_cos = F.cosine_similarity(achieved, target_activation, dim=-1).cpu().tolist()

    return {
        "prefix": prefix.detach().cpu(),
        "final_cosine": final_cos,
        "per_position_cosine": per_pos_cos,
        "cosine_history": cosine_history,
    }


def optimize_prefix_with_restarts(context_embeds, target_activation, target_positions,
                                   n_restarts=N_RESTARTS, **kwargs):
    """Multiple restarts, return best."""
    best = None
    all_cosines = []
    for i in range(n_restarts):
        result = optimize_prefix(context_embeds, target_activation, target_positions,
                                  seed=i * 1000, **kwargs)
        all_cosines.append(result["final_cosine"])
        if best is None or result["final_cosine"] > best["final_cosine"]:
            best = result
    best["all_restart_cosines"] = all_cosines
    return best


# ── Load contexts ───────────────────────────────────────────────────

print("=== Prefix Soft Prompt Steering Experiment ===\n")

# Load longer sequences
print("Loading contexts...")
dataset_sequences = []
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
for example in dataset:
    text = example["text"].strip()
    if len(text) < 50:
        continue
    tokens = tokenizer.encode(text, return_tensors="pt").squeeze(0)
    if tokens.shape[0] >= CONTEXT_LEN:
        dataset_sequences.append(tokens[:CONTEXT_LEN].to(DEVICE))
    if len(dataset_sequences) >= 50:
        break
print(f"Loaded {len(dataset_sequences)} contexts of length {CONTEXT_LEN}")

# ── Experiment A: Steer LAST position ──────────────────────────────

print("\n--- Experiment A: Steer last position (pos 119) ---")
print(f"  {PREFIX_LEN}-token prefix + {CONTEXT_LEN}-token context")
print(f"  Target: layer {TARGET_LAYER} activation at position {TOTAL_LEN - 1}")

N_TARGETS_A = 20
results_last = []
start_time = time.time()

for ti in range(N_TARGETS_A):
    context_tokens = dataset_sequences[ti]
    context_embeds = model.gpt_neox.embed_in(context_tokens.unsqueeze(0)).squeeze(0).detach()

    # Get the "natural" activation at position 119 (without any prefix modification)
    # as the target — we'll use a DIFFERENT context's activation as the target
    target_context = dataset_sequences[(ti + 10) % len(dataset_sequences)]
    target_embeds = model.gpt_neox.embed_in(target_context.unsqueeze(0)).squeeze(0).detach()
    # Compute target activation for that context at its last position
    with torch.no_grad():
        # Use zero prefix + target context
        zero_prefix = torch.zeros(PREFIX_LEN, D_MODEL, device=DEVICE)
        full_target = torch.cat([zero_prefix, target_embeds], dim=0)
        target_residual = get_residual_fast_long(full_target)
        target_activation = target_residual[TOTAL_LEN - 1:TOTAL_LEN]  # [1, 768]

    result = optimize_prefix_with_restarts(
        context_embeds, target_activation,
        target_positions=[TOTAL_LEN - 1],
    )
    results_last.append(result)

    if (ti + 1) % 5 == 0:
        elapsed = time.time() - start_time
        median_cos = np.median([r["final_cosine"] for r in results_last])
        print(f"  Target {ti+1}/{N_TARGETS_A}, median_cos={median_cos:.4f}, elapsed: {elapsed:.0f}s")

cosines_last = [r["final_cosine"] for r in results_last]
print(f"\n  Last-position steering results:")
print(f"  Median cosine: {np.median(cosines_last):.4f}")
print(f"  Mean cosine: {np.mean(cosines_last):.4f}")
print(f"  Min: {np.min(cosines_last):.4f}, Max: {np.max(cosines_last):.4f}")

# ── Experiment B: Steer MULTIPLE positions ─────────────────────────

print("\n--- Experiment B: Steer multiple positions simultaneously ---")

position_configs = {
    "last_1": [TOTAL_LEN - 1],
    "last_5": list(range(TOTAL_LEN - 5, TOTAL_LEN)),
    "last_10": list(range(TOTAL_LEN - 10, TOTAL_LEN)),
    "last_20": list(range(TOTAL_LEN - 20, TOTAL_LEN)),
    "all_context": list(range(PREFIX_LEN, TOTAL_LEN)),  # all 100 context positions
}

N_TARGETS_B = 10
results_multi = {}

for config_name, target_positions in position_configs.items():
    n_pos = len(target_positions)
    print(f"\n  Config '{config_name}': steering {n_pos} positions")
    config_results = []

    for ti in range(N_TARGETS_B):
        context_tokens = dataset_sequences[ti]
        context_embeds = model.gpt_neox.embed_in(context_tokens.unsqueeze(0)).squeeze(0).detach()

        # Target: different context's activations at the same positions
        target_context = dataset_sequences[(ti + 10) % len(dataset_sequences)]
        target_embeds_t = model.gpt_neox.embed_in(target_context.unsqueeze(0)).squeeze(0).detach()
        with torch.no_grad():
            zero_prefix = torch.zeros(PREFIX_LEN, D_MODEL, device=DEVICE)
            full_target = torch.cat([zero_prefix, target_embeds_t], dim=0)
            target_residual = get_residual_fast_long(full_target)
            target_activation = target_residual[target_positions]  # [n_pos, 768]

        result = optimize_prefix_with_restarts(
            context_embeds, target_activation,
            target_positions=target_positions,
        )
        config_results.append(result)

    cosines = [r["final_cosine"] for r in config_results]
    results_multi[config_name] = {
        "n_positions": n_pos,
        "median_cosine": float(np.median(cosines)),
        "mean_cosine": float(np.mean(cosines)),
        "std_cosine": float(np.std(cosines)),
        "all_cosines": cosines,
    }
    print(f"    Median cos: {np.median(cosines):.4f}, Mean: {np.mean(cosines):.4f}")

# ── Experiment C: Effect of position distance ──────────────────────

print("\n--- Experiment C: How does steering quality vary with distance from prefix? ---")

N_TARGETS_C = 10
# Steer individual positions at different distances from the prefix
distance_positions = [PREFIX_LEN + d for d in [0, 5, 10, 20, 40, 60, 80, 99]]
results_distance = {d: [] for d in distance_positions}

for ti in range(N_TARGETS_C):
    context_tokens = dataset_sequences[ti]
    context_embeds = model.gpt_neox.embed_in(context_tokens.unsqueeze(0)).squeeze(0).detach()
    target_context = dataset_sequences[(ti + 10) % len(dataset_sequences)]
    target_embeds_t = model.gpt_neox.embed_in(target_context.unsqueeze(0)).squeeze(0).detach()

    for pos in distance_positions:
        with torch.no_grad():
            zero_prefix = torch.zeros(PREFIX_LEN, D_MODEL, device=DEVICE)
            full_target = torch.cat([zero_prefix, target_embeds_t], dim=0)
            target_residual = get_residual_fast_long(full_target)
            target_activation = target_residual[pos:pos+1]

        result = optimize_prefix_with_restarts(
            context_embeds, target_activation,
            target_positions=[pos],
        )
        results_distance[pos].append(result["final_cosine"])

    if (ti + 1) % 5 == 0:
        elapsed = time.time() - start_time
        print(f"  Target {ti+1}/{N_TARGETS_C}, elapsed: {elapsed:.0f}s")

# ── Save results ────────────────────────────────────────────────────

save_data = {
    "experiment_A_last_position": {
        "median_cosine": float(np.median(cosines_last)),
        "mean_cosine": float(np.mean(cosines_last)),
        "all_cosines": cosines_last,
    },
    "experiment_B_multi_position": results_multi,
    "experiment_C_distance": {
        str(pos): {
            "distance_from_prefix": pos - PREFIX_LEN,
            "median_cosine": float(np.median(results_distance[pos])),
            "mean_cosine": float(np.mean(results_distance[pos])),
            "all_cosines": results_distance[pos],
        }
        for pos in distance_positions
    },
}

with open(RESULTS_DIR / "prefix_steering.json", "w") as f:
    json.dump(save_data, f, indent=2)
print("\nSaved results/prefix_steering.json")

# ── Plot ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot A: Last-position histogram
ax = axes[0]
ax.hist(cosines_last, bins=15, edgecolor='black', alpha=0.7, color='#2196F3')
ax.axvline(np.median(cosines_last), color='red', linestyle='--',
           label=f'Median: {np.median(cosines_last):.4f}')
ax.set_xlabel('Final Cosine Similarity')
ax.set_ylabel('Count')
ax.set_title(f'A: Steer Position {TOTAL_LEN-1}\n({PREFIX_LEN}-token prefix, {CONTEXT_LEN}-token context)')
ax.legend()

# Plot B: Multi-position comparison
ax = axes[1]
configs = sorted(results_multi.keys(), key=lambda k: results_multi[k]["n_positions"])
x = range(len(configs))
medians = [results_multi[c]["median_cosine"] for c in configs]
means = [results_multi[c]["mean_cosine"] for c in configs]
stds = [results_multi[c]["std_cosine"] for c in configs]
n_positions = [results_multi[c]["n_positions"] for c in configs]
ax.errorbar(x, means, yerr=stds, fmt='bo-', capsize=5, linewidth=2, markersize=8)
ax.plot(x, medians, 'rs--', markersize=8, label='Median')
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(configs, n_positions)], fontsize=8)
ax.set_ylabel('Cosine Similarity')
ax.set_title('B: Steering Quality vs # Positions')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot C: Distance from prefix
ax = axes[2]
distances = [pos - PREFIX_LEN for pos in distance_positions]
median_by_dist = [np.median(results_distance[pos]) for pos in distance_positions]
mean_by_dist = [np.mean(results_distance[pos]) for pos in distance_positions]
for pos in distance_positions:
    d = pos - PREFIX_LEN
    vals = results_distance[pos]
    ax.scatter([d] * len(vals), vals, alpha=0.3, s=20, color='blue')
ax.plot(distances, median_by_dist, 'ro-', linewidth=2, markersize=8, zorder=5, label='Median')
ax.set_xlabel('Distance from Prefix (tokens)')
ax.set_ylabel('Cosine Similarity')
ax.set_title('C: Steering Quality vs Distance')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Prefix Soft Prompt Steering: Can 20 Tokens Steer a 100-Token Context?', fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "prefix_steering.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved results/prefix_steering.png")

# Print summary
print("\n=== Summary ===")
print(f"\nA. Steer last position (pos {TOTAL_LEN-1}):")
print(f"   Median cosine: {np.median(cosines_last):.4f}")

print(f"\nB. Multi-position steering:")
for c in configs:
    d = results_multi[c]
    print(f"   {c:>15}: median={d['median_cosine']:.4f}, n_pos={d['n_positions']}")

print(f"\nC. Distance from prefix:")
for pos in distance_positions:
    d = pos - PREFIX_LEN
    m = np.median(results_distance[pos])
    print(f"   Distance {d:>3}: median_cos={m:.4f}")

total_time = time.time() - start_time
print(f"\nTotal time: {total_time:.0f}s")
