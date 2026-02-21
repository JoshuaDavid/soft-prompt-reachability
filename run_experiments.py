"""
Soft Prompt Reachability Experiments
====================================
Investigates whether arbitrary residual stream activations at intermediate
layers of Pythia-160M are reachable via gradient-optimized layer-0 inputs.

See RESEARCH_AGENDA.md for full motivation and design.
"""

import json
import os
import time
import sys
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Configuration ──────────────────────────────────────────────────────

MODEL_NAME = "EleutherAI/pythia-160m"
TARGET_LAYER = 6  # middle of 12-layer model
SEQ_LEN = 20
RESULTS_DIR = Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimization defaults
DEFAULT_STEPS = 2000
DEFAULT_LR = 0.01
DEFAULT_RESTARTS = 5
MSE_WEIGHT = 0.01

print(f"Device: {DEVICE}")
print(f"Target layer: {TARGET_LAYER}, Sequence length: {SEQ_LEN}")


# ── Model Setup ────────────────────────────────────────────────────────

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()
for param in model.parameters():
    param.requires_grad = False
model = model.to(DEVICE)

D_MODEL = model.config.hidden_size
NUM_LAYERS = model.config.num_hidden_layers
print(f"Model loaded: {NUM_LAYERS} layers, d_model={D_MODEL}")


# ── Core Functions ─────────────────────────────────────────────────────

def get_all_residuals(input_embeds):
    """
    Run input embeddings through model, return residual stream at ALL layers.

    Args:
        input_embeds: Tensor of shape [SEQ_LEN, D_MODEL]
    Returns:
        List of Tensors, each [SEQ_LEN, D_MODEL], from layer 0 to final layer
    """
    outputs = model(
        inputs_embeds=input_embeds.unsqueeze(0),
        output_hidden_states=True,
    )
    # hidden_states: tuple of (num_layers + 1) tensors
    # [0] = embedding output, [1] = after layer 0, ..., [12] = after layer 11
    return [h.squeeze(0) for h in outputs.hidden_states]


def get_residual(input_embeds, layer=TARGET_LAYER):
    """Convenience wrapper: single layer's residual."""
    return get_all_residuals(input_embeds)[layer]


def get_real_target(token_ids):
    """Get the residual stream for a real token sequence."""
    with torch.no_grad():
        embeds = model.gpt_neox.embed_in(token_ids.unsqueeze(0)).squeeze(0)
        return get_residual(embeds).detach()


def get_real_all_residuals(token_ids):
    """Get all-layer residual streams for a real token sequence."""
    with torch.no_grad():
        embeds = model.gpt_neox.embed_in(token_ids.unsqueeze(0)).squeeze(0)
        return [r.detach() for r in get_all_residuals(embeds)]


def cosine_sim_per_position(achieved, target):
    """Per-position cosine similarity, returns [SEQ_LEN] tensor."""
    return F.cosine_similarity(achieved, target, dim=-1)


def cosine_loss(achieved, target):
    """1 - mean per-position cosine similarity."""
    return 1 - cosine_sim_per_position(achieved, target).mean()


def combined_loss(achieved, target, mse_weight=MSE_WEIGHT):
    """Primary: cosine similarity. Secondary: MSE."""
    cos = cosine_loss(achieved, target)
    mse = ((achieved - target) ** 2).mean()
    return cos + mse_weight * mse


def optimize_prompt(target, steps=DEFAULT_STEPS, lr=DEFAULT_LR, seed=None,
                    get_residual_fn=None, log_every=200):
    """
    Optimize a soft prompt to hit target residual stream activations.
    """
    if get_residual_fn is None:
        get_residual_fn = lambda x: get_residual(x)

    if seed is not None:
        torch.manual_seed(seed)

    prompt = torch.randn(SEQ_LEN, D_MODEL, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([prompt], lr=lr)

    loss_history, cosine_history, mse_history = [], [], []
    for step in range(steps):
        optimizer.zero_grad()
        residual = get_residual_fn(prompt)
        loss = combined_loss(residual, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cos = cosine_sim_per_position(residual, target).mean().item()
            mse = ((residual - target) ** 2).mean().item()

        loss_history.append(loss.item())
        cosine_history.append(cos)
        mse_history.append(mse)

        if log_every and step % log_every == 0:
            print(f"  Step {step:4d}: loss={loss.item():.6f}, cos={cos:.4f}, mse={mse:.4f}")

    # Record all-layer residuals at final state
    with torch.no_grad():
        all_residuals = get_all_residuals(prompt.detach())

    return {
        "prompt": prompt.detach(),
        "loss_history": loss_history,
        "cosine_history": cosine_history,
        "mse_history": mse_history,
        "all_layer_residuals": [r.detach().cpu() for r in all_residuals],
        "final_cosine": cosine_history[-1],
        "final_mse": mse_history[-1],
    }


def optimize_with_restarts(target, n_restarts=DEFAULT_RESTARTS, **kwargs):
    """Run multiple random restarts and return the best result."""
    best_result = None
    best_final = float("inf")
    all_final_losses = []
    all_final_cosines = []

    for i in range(n_restarts):
        result = optimize_prompt(target, seed=i * 1000, **kwargs)
        final = result["loss_history"][-1]
        all_final_losses.append(final)
        all_final_cosines.append(result["final_cosine"])
        if final < best_final:
            best_final = final
            best_result = result

    return {
        "best": best_result,
        "all_final_losses": all_final_losses,
        "all_final_cosines": all_final_cosines,
        "restart_variance": float(np.var(all_final_losses)),
    }


# ── Data Loading ───────────────────────────────────────────────────────

def load_real_sequences(n=100):
    """Load n real token sequences of length SEQ_LEN from wikitext-2."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    sequences = []
    for example in dataset:
        text = example["text"].strip()
        if len(text) < 20:
            continue
        tokens = tokenizer.encode(text, return_tensors="pt").squeeze(0)
        if tokens.shape[0] >= SEQ_LEN:
            sequences.append(tokens[:SEQ_LEN].to(DEVICE))
        if len(sequences) >= n:
            break
    return sequences


# ── Embedding Manifold Distance (Experiment 1b) ───────────────────────

def embedding_manifold_distances(optimized_prompts):
    """Compute distance from optimized inputs to nearest token embeddings."""
    embed_matrix = model.gpt_neox.embed_in.weight.detach()  # [vocab, D_MODEL]
    embed_normed = embed_matrix / embed_matrix.norm(dim=-1, keepdim=True)

    all_cos_sims = []
    all_l2_dists = []
    all_nearest_tokens = []

    for prompt in optimized_prompts:
        prompt_dev = prompt.to(DEVICE)
        prompt_normed = prompt_dev / prompt_dev.norm(dim=-1, keepdim=True)
        cos_sims = prompt_normed @ embed_normed.T  # [SEQ_LEN, vocab]
        best_cos, best_idx = cos_sims.max(dim=-1)
        best_embeds = embed_matrix[best_idx]
        l2_dists = (prompt_dev - best_embeds).norm(dim=-1)

        all_cos_sims.append(best_cos.cpu().numpy())
        all_l2_dists.append(l2_dists.cpu().numpy())
        all_nearest_tokens.append(best_idx.cpu().numpy())

    return {
        "cosine_to_nearest": all_cos_sims,
        "l2_to_nearest": all_l2_dists,
        "nearest_token_ids": all_nearest_tokens,
    }


# ── Ablation Hooks (Experiment 4) ─────────────────────────────────────

def install_ablation_hooks(ablate="mlp"):
    """
    Install hooks to zero out either MLP or attention outputs.

    Pythia uses parallel residual: x = x + attn(ln1(x)) + mlp(ln2(x))
    We hook the submodule to return zeros, effectively removing its contribution.

    Args:
        ablate: "mlp" to ablate FFN, "attention" to ablate attention
    Returns:
        list of hook handles (call .remove() to undo)
    """
    hooks = []
    for layer in model.gpt_neox.layers:
        if ablate == "mlp":
            target_module = layer.mlp
        elif ablate == "attention":
            target_module = layer.attention
        else:
            raise ValueError(f"Unknown ablation target: {ablate}")

        def make_hook(mod_type):
            def hook_fn(module, input, output):
                if mod_type == "attention":
                    # Attention returns (attn_output, present)
                    return (torch.zeros_like(output[0]),) + output[1:]
                else:
                    # MLP returns a single tensor
                    return torch.zeros_like(output)
            return hook_fn

        h = target_module.register_forward_hook(make_hook(ablate))
        hooks.append(h)
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ── Plotting Utilities ─────────────────────────────────────────────────

def plot_loss_curves(all_results, save_path, title="Loss Curves"):
    """Plot overlaid loss curves with median highlighted."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    cosine_curves = [r["best"]["cosine_history"] for r in all_results]
    mse_curves = [r["best"]["mse_history"] for r in all_results]

    for cos_h in cosine_curves:
        ax1.plot(cos_h, color='blue', alpha=0.1, linewidth=0.5)
    for mse_h in mse_curves:
        ax2.plot(mse_h, color='red', alpha=0.1, linewidth=0.5)

    # Median
    cos_array = np.array(cosine_curves)
    mse_array = np.array(mse_curves)
    ax1.plot(np.median(cos_array, axis=0), color='blue', linewidth=2, label='Median Cosine Sim')
    ax2.plot(np.median(mse_array, axis=0), color='red', linewidth=2, label='Median MSE')

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cosine Similarity', color='blue')
    ax2.set_ylabel('MSE', color='red')
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cosine_histogram(all_results, save_path, title="Final Cosine Similarity"):
    """Histogram of best-of-restarts final cosine similarity."""
    finals = [r["best"]["final_cosine"] for r in all_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(finals, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Final Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.axvline(np.median(finals), color='red', linestyle='--', label=f'Median: {np.median(finals):.4f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cosine_by_position(all_results, targets, save_path, title="Cosine Similarity by Position"):
    """Per-position cosine similarity with error bars."""
    per_pos = []
    for r, t in zip(all_results, targets):
        prompt = r["best"]["prompt"].to(DEVICE)
        with torch.no_grad():
            achieved = get_residual(prompt)
            cos_per_pos = cosine_sim_per_position(achieved, t).cpu().numpy()
        per_pos.append(cos_per_pos)

    per_pos = np.array(per_pos)  # [n_targets, SEQ_LEN]
    means = per_pos.mean(axis=0)
    stds = per_pos.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(SEQ_LEN)
    ax.errorbar(positions, means, yerr=stds, fmt='o-', capsize=3)
    ax.set_xlabel('Position')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(title)
    ax.set_xticks(positions)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_layer_trajectories(results, targets, real_all_residuals_list, save_path,
                            title="Layer Trajectories"):
    """
    Plot per-layer cosine similarity for best, median, worst cases.
    Compares optimized input's trajectory vs real input's trajectory.
    """
    finals = [r["best"]["final_cosine"] for r in results]
    sorted_idx = np.argsort(finals)

    cases = {
        "Best": sorted_idx[-1],
        "Median": sorted_idx[len(sorted_idx) // 2],
        "Worst": sorted_idx[0],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (label, idx) in zip(axes, cases.items()):
        opt_residuals = results[idx]["best"]["all_layer_residuals"]
        real_residuals = real_all_residuals_list[idx]
        target = targets[idx]

        # Cosine sim between optimized trajectory and target at each layer
        opt_cos_per_layer = []
        real_cos_per_layer = []
        for layer_idx in range(len(opt_residuals)):
            opt_r = opt_residuals[layer_idx].to(DEVICE)
            real_r = real_residuals[layer_idx].to(DEVICE)
            opt_cos = cosine_sim_per_position(opt_r, target).mean().item()
            real_cos = cosine_sim_per_position(real_r, target).mean().item()
            opt_cos_per_layer.append(opt_cos)
            real_cos_per_layer.append(real_cos)

        layers = list(range(len(opt_residuals)))
        ax.plot(layers, opt_cos_per_layer, 'b-o', label='Optimized', markersize=4)
        ax.plot(layers, real_cos_per_layer, 'r-s', label='Real', markersize=4)
        ax.axvline(TARGET_LAYER, color='gray', linestyle='--', alpha=0.5, label=f'Target Layer ({TARGET_LAYER})')
        ax.set_xlabel('Layer')
        ax.set_title(f'{label} (final cos={finals[idx]:.4f})')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Cosine Sim to Target')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_metrics(all_results, save_path, extra=None):
    """Save raw metrics to JSON."""
    metrics = {
        "n_targets": len(all_results),
        "final_cosines": [r["best"]["final_cosine"] for r in all_results],
        "final_mses": [r["best"]["final_mse"] for r in all_results],
        "restart_variances": [r["restart_variance"] for r in all_results],
        "all_restart_cosines": [r["all_final_cosines"] for r in all_results],
        "summary": {
            "mean_cosine": float(np.mean([r["best"]["final_cosine"] for r in all_results])),
            "median_cosine": float(np.median([r["best"]["final_cosine"] for r in all_results])),
            "std_cosine": float(np.std([r["best"]["final_cosine"] for r in all_results])),
            "min_cosine": float(np.min([r["best"]["final_cosine"] for r in all_results])),
            "max_cosine": float(np.max([r["best"]["final_cosine"] for r in all_results])),
            "mean_mse": float(np.mean([r["best"]["final_mse"] for r in all_results])),
            "median_mse": float(np.median([r["best"]["final_mse"] for r in all_results])),
        },
    }
    if extra:
        metrics.update(extra)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {save_path}")


# ── Experiment 1: Reachability of Real Activations ─────────────────────

def run_experiment1(n_targets=100, steps=DEFAULT_STEPS, n_restarts=DEFAULT_RESTARTS):
    """Run Experiment 1: optimize inputs to match real activation targets."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Reachability of Real Activations")
    print("="*70)

    exp_dir = RESULTS_DIR / "experiment1"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences and compute targets
    print("Loading real sequences...")
    sequences = load_real_sequences(n_targets)
    print(f"Loaded {len(sequences)} sequences")

    targets = []
    real_all_residuals_list = []
    print("Computing target activations...")
    for i, seq in enumerate(sequences):
        target = get_real_target(seq)
        targets.append(target)
        all_res = get_real_all_residuals(seq)
        real_all_residuals_list.append([r.cpu() for r in all_res])
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(sequences)} targets computed")

    # Run optimization
    all_results = []
    t0 = time.time()
    for i, target in enumerate(targets):
        print(f"\nTarget {i+1}/{len(targets)}:")
        result = optimize_with_restarts(target, n_restarts=n_restarts, steps=steps)
        all_results.append(result)
        elapsed = time.time() - t0
        per_target = elapsed / (i + 1)
        remaining = per_target * (len(targets) - i - 1)
        print(f"  Best cosine: {result['best']['final_cosine']:.4f}, "
              f"Restart var: {result['restart_variance']:.6f}")
        print(f"  Elapsed: {elapsed:.0f}s, ETA: {remaining:.0f}s")

        # Checkpoint every 10 targets
        if (i + 1) % 10 == 0:
            save_metrics(all_results, exp_dir / "metrics_checkpoint.json")

    # Generate plots and metrics
    print("\nGenerating Experiment 1 outputs...")
    plot_loss_curves(all_results, exp_dir / "loss_curves.png",
                     "Experiment 1: Loss Curves (Real Targets)")
    plot_cosine_histogram(all_results, exp_dir / "final_cosine_histogram.png",
                          "Experiment 1: Final Cosine Similarity (Real Targets)")
    plot_cosine_by_position(all_results, targets, exp_dir / "cosine_by_position.png",
                            "Experiment 1: Cosine Similarity by Position")
    plot_layer_trajectories(all_results, targets, real_all_residuals_list,
                            exp_dir / "layer_trajectories.png",
                            "Experiment 1: Layer Trajectories")
    save_metrics(all_results, exp_dir / "metrics.json")

    # Check if 2000 steps is enough
    median_cos_at_end = np.median([r["best"]["cosine_history"][-1] for r in all_results])
    median_cos_at_80pct = np.median([r["best"]["cosine_history"][int(steps*0.8)] for r in all_results])
    improvement = median_cos_at_end - median_cos_at_80pct
    print(f"\nMedian cosine improvement in last 20% of steps: {improvement:.6f}")
    if improvement > 0.01:
        print("WARNING: Still improving significantly - consider extending steps")

    return all_results, targets, real_all_residuals_list, sequences


# ── Experiment 1b: Embedding Manifold Distance ─────────────────────────

def run_experiment1b(exp1_results):
    """Analyze distance from optimized inputs to token embedding manifold."""
    print("\n" + "="*70)
    print("EXPERIMENT 1b: Embedding Manifold Distance")
    print("="*70)

    exp_dir = RESULTS_DIR / "experiment1b"
    exp_dir.mkdir(parents=True, exist_ok=True)

    optimized_prompts = [r["best"]["prompt"] for r in exp1_results]
    distances = embedding_manifold_distances(optimized_prompts)

    # Plot histogram of cosine similarity to nearest token embedding
    all_cos = np.concatenate(distances["cosine_to_nearest"])
    all_l2 = np.concatenate(distances["l2_to_nearest"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(all_cos, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Cosine Similarity to Nearest Token Embedding')
    ax1.set_ylabel('Count')
    ax1.set_title('Optimized Input vs Nearest Token Embedding')
    ax1.axvline(np.median(all_cos), color='red', linestyle='--',
                label=f'Median: {np.median(all_cos):.4f}')
    ax1.legend()

    ax2.hist(all_l2, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel('L2 Distance to Nearest Token Embedding')
    ax2.set_ylabel('Count')
    ax2.set_title('L2 Distance Distribution')
    ax2.axvline(np.median(all_l2), color='red', linestyle='--',
                label=f'Median: {np.median(all_l2):.4f}')
    ax2.legend()

    fig.suptitle('Experiment 1b: Embedding Manifold Distance')
    fig.tight_layout()
    fig.savefig(exp_dir / "embedding_distance.png", dpi=150, bbox_inches='tight')
    # Also save to top-level results
    fig.savefig(RESULTS_DIR / "experiment1b_embedding_distance.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved embedding distance plots")

    # Also compute: what fraction of optimized vectors have cos > 0.9 to a real token?
    frac_close = (all_cos > 0.9).mean()
    frac_very_close = (all_cos > 0.95).mean()

    # Compute embedding matrix stats for reference
    embed_matrix = model.gpt_neox.embed_in.weight.detach()
    embed_norms = embed_matrix.norm(dim=-1)

    # PCA distance: project optimized vectors and embedding matrix into PCA space
    # Use top-k PCA components of embedding matrix as reference
    embed_centered = embed_matrix - embed_matrix.mean(dim=0)
    U, S, Vh = torch.linalg.svd(embed_centered, full_matrices=False)

    # Explained variance
    explained_var = (S ** 2) / (S ** 2).sum()
    cumulative_var = explained_var.cumsum(0).cpu().numpy()

    # For various k, compute reconstruction error
    pca_analysis = {}
    for k in [10, 50, 100, 200, 384, 768]:
        if k > D_MODEL:
            continue
        basis = Vh[:k]  # [k, D_MODEL]
        all_recon_errors = []
        for prompt in optimized_prompts:
            prompt_dev = prompt.to(DEVICE)
            prompt_centered = prompt_dev - embed_matrix.mean(dim=0)
            projection = prompt_centered @ basis.T @ basis
            recon_error = (prompt_centered - projection).norm(dim=-1).cpu().numpy()
            all_recon_errors.append(recon_error)
        all_recon_errors = np.concatenate(all_recon_errors)
        pca_analysis[f"k={k}"] = {
            "mean_recon_error": float(all_recon_errors.mean()),
            "median_recon_error": float(np.median(all_recon_errors)),
            "cumulative_variance_explained": float(cumulative_var[k-1]),
        }

    metrics = {
        "cosine_to_nearest": {
            "mean": float(all_cos.mean()),
            "median": float(np.median(all_cos)),
            "std": float(all_cos.std()),
            "min": float(all_cos.min()),
            "max": float(all_cos.max()),
            "frac_above_0.9": float(frac_close),
            "frac_above_0.95": float(frac_very_close),
        },
        "l2_to_nearest": {
            "mean": float(all_l2.mean()),
            "median": float(np.median(all_l2)),
            "std": float(all_l2.std()),
        },
        "embedding_norms": {
            "mean": float(embed_norms.mean().item()),
            "std": float(embed_norms.std().item()),
        },
        "pca_analysis": pca_analysis,
    }
    with open(exp_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics")

    return metrics


# ── Experiment 4: FFN and Attention Ablation ───────────────────────────

def run_experiment4(targets, steps=DEFAULT_STEPS, n_restarts=DEFAULT_RESTARTS,
                    exp1_results=None):
    """Run Experiment 4: ablation study."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: FFN and Attention Ablation")
    print("="*70)

    exp_dir = RESULTS_DIR / "experiment4"
    exp_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 4c: Full model (control) - reuse Exp1 results if available
    if exp1_results is not None:
        results["full"] = exp1_results
        print("Using Experiment 1 results as full-model control")
    else:
        print("\n--- 4c: Full Model (Control) ---")
        results["full"] = []
        for i, target in enumerate(targets):
            print(f"  Full model target {i+1}/{len(targets)}:")
            r = optimize_with_restarts(target, n_restarts=n_restarts, steps=steps)
            results["full"].append(r)

    # 4a: FFN Ablated (Attention Only)
    print("\n--- 4a: FFN Ablated (Attention Only) ---")
    hooks = install_ablation_hooks(ablate="mlp")

    def get_residual_no_ffn(x):
        return get_residual(x)

    results["no_ffn"] = []
    t0 = time.time()
    for i, target in enumerate(targets):
        print(f"  No-FFN target {i+1}/{len(targets)}:")
        r = optimize_with_restarts(target, n_restarts=n_restarts, steps=steps,
                                   get_residual_fn=get_residual_no_ffn)
        results["no_ffn"].append(r)
        elapsed = time.time() - t0
        per_target = elapsed / (i + 1)
        remaining = per_target * (len(targets) - i - 1)
        print(f"  Best cosine: {r['best']['final_cosine']:.4f}, ETA: {remaining:.0f}s")

        if (i + 1) % 10 == 0:
            save_metrics(results["no_ffn"], exp_dir / "no_ffn_checkpoint.json")

    remove_hooks(hooks)

    # 4b: Attention Ablated (FFN Only)
    print("\n--- 4b: Attention Ablated (FFN Only) ---")
    hooks = install_ablation_hooks(ablate="attention")

    def get_residual_no_attn(x):
        return get_residual(x)

    results["no_attn"] = []
    t0 = time.time()
    for i, target in enumerate(targets):
        print(f"  No-Attn target {i+1}/{len(targets)}:")
        r = optimize_with_restarts(target, n_restarts=n_restarts, steps=steps,
                                   get_residual_fn=get_residual_no_attn)
        results["no_attn"].append(r)
        elapsed = time.time() - t0
        per_target = elapsed / (i + 1)
        remaining = per_target * (len(targets) - i - 1)
        print(f"  Best cosine: {r['best']['final_cosine']:.4f}, ETA: {remaining:.0f}s")

        if (i + 1) % 10 == 0:
            save_metrics(results["no_attn"], exp_dir / "no_attn_checkpoint.json")

    remove_hooks(hooks)

    # Generate comparison plots
    print("\nGenerating Experiment 4 outputs...")

    # Save individual metrics
    for variant in ["no_ffn", "no_attn"]:
        save_metrics(results[variant], exp_dir / f"metrics_{variant}.json")

    # Plot: ablation comparison scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    full_cos = [r["best"]["final_cosine"] for r in results["full"]]
    noffn_cos = [r["best"]["final_cosine"] for r in results["no_ffn"]]
    noattn_cos = [r["best"]["final_cosine"] for r in results["no_attn"]]

    ax1.scatter(full_cos, noffn_cos, alpha=0.5, s=20)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel('Full Model Cosine Sim')
    ax1.set_ylabel('No-FFN Cosine Sim')
    ax1.set_title('4a: FFN Ablated vs Full')

    ax2.scatter(full_cos, noattn_cos, alpha=0.5, s=20, color='orange')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('Full Model Cosine Sim')
    ax2.set_ylabel('No-Attention Cosine Sim')
    ax2.set_title('4b: Attention Ablated vs Full')

    fig.suptitle('Experiment 4: Ablation Comparison')
    fig.tight_layout()
    fig.savefig(exp_dir / "ablation_comparison.png", dpi=150, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / "ablation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Also plot histograms overlaid
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(full_cos, bins=30, alpha=0.5, label=f'Full (median={np.median(full_cos):.4f})', color='blue')
    ax.hist(noffn_cos, bins=30, alpha=0.5, label=f'No-FFN (median={np.median(noffn_cos):.4f})', color='green')
    ax.hist(noattn_cos, bins=30, alpha=0.5, label=f'No-Attn (median={np.median(noattn_cos):.4f})', color='orange')
    ax.set_xlabel('Final Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Experiment 4: Ablation Comparison Histograms')
    ax.legend()
    fig.tight_layout()
    fig.savefig(exp_dir / "ablation_histograms.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Per-target delta analysis
    delta_noffn = np.array(noffn_cos) - np.array(full_cos)
    delta_noattn = np.array(noattn_cos) - np.array(full_cos)

    ablation_summary = {
        "full_model": {"mean": float(np.mean(full_cos)), "median": float(np.median(full_cos))},
        "no_ffn": {
            "mean": float(np.mean(noffn_cos)), "median": float(np.median(noffn_cos)),
            "mean_delta": float(delta_noffn.mean()),
            "frac_better": float((delta_noffn > 0).mean()),
        },
        "no_attn": {
            "mean": float(np.mean(noattn_cos)), "median": float(np.median(noattn_cos)),
            "mean_delta": float(delta_noattn.mean()),
            "frac_better": float((delta_noattn > 0).mean()),
        },
    }
    with open(exp_dir / "ablation_summary.json", 'w') as f:
        json.dump(ablation_summary, f, indent=2)
    print(f"  Ablation summary: {json.dumps(ablation_summary, indent=2)}")

    return results


# ── Experiment 2: Random Targets ───────────────────────────────────────

def compute_activation_statistics(n_sequences=1000):
    """Compute per-position mean and covariance of layer-6 residuals."""
    print("Computing activation statistics from real data...")
    sequences = load_real_sequences(n_sequences)

    all_residuals = []
    for i, seq in enumerate(sequences):
        target = get_real_target(seq)
        all_residuals.append(target.cpu())
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sequences)} sequences processed")

    # Stack: [n_sequences, SEQ_LEN, D_MODEL]
    stacked = torch.stack(all_residuals)

    # Per-position statistics
    per_pos_mean = stacked.mean(dim=0)  # [SEQ_LEN, D_MODEL]
    per_pos_norms = stacked.norm(dim=-1)  # [n_sequences, SEQ_LEN]
    typical_norms = per_pos_norms.mean(dim=0)  # [SEQ_LEN]

    # Per-position covariance (for distribution-matched sampling)
    per_pos_cov = []
    for pos in range(SEQ_LEN):
        data = stacked[:, pos, :]  # [n_sequences, D_MODEL]
        cov = torch.cov(data.T)  # [D_MODEL, D_MODEL]
        per_pos_cov.append(cov)

    return {
        "mean": per_pos_mean,
        "covariances": per_pos_cov,
        "typical_norms": typical_norms,
        "all_residuals": stacked,
    }


def run_experiment2a(stats, n_targets=100, steps=DEFAULT_STEPS, n_restarts=DEFAULT_RESTARTS):
    """Experiment 2a: Distribution-matched random targets."""
    print("\n" + "="*70)
    print("EXPERIMENT 2a: Distribution-Matched Random Targets")
    print("="*70)

    exp_dir = RESULTS_DIR / "experiment2a"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Generate targets from per-position Gaussian
    torch.manual_seed(42)
    targets = []
    for _ in range(n_targets):
        target_positions = []
        for pos in range(SEQ_LEN):
            mu = stats["mean"][pos]
            cov = stats["covariances"][pos]
            # Sample from N(mu, cov) via Cholesky
            try:
                L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(D_MODEL))
                z = torch.randn(D_MODEL)
                sample = mu + L @ z
            except Exception:
                # Fallback: diagonal covariance
                var = cov.diag().clamp(min=1e-6)
                sample = mu + torch.randn(D_MODEL) * var.sqrt()
            target_positions.append(sample)
        target = torch.stack(target_positions).to(DEVICE)
        targets.append(target)

    # Optimize
    all_results = []
    t0 = time.time()
    for i, target in enumerate(targets):
        print(f"\n2a Target {i+1}/{len(targets)}:")
        r = optimize_with_restarts(target, n_restarts=n_restarts, steps=steps)
        all_results.append(r)
        elapsed = time.time() - t0
        per_target = elapsed / (i + 1)
        remaining = per_target * (len(targets) - i - 1)
        print(f"  Best cosine: {r['best']['final_cosine']:.4f}, ETA: {remaining:.0f}s")

        if (i + 1) % 10 == 0:
            save_metrics(all_results, exp_dir / "metrics_checkpoint.json")

    # Generate outputs
    print("\nGenerating Experiment 2a outputs...")
    plot_loss_curves(all_results, exp_dir / "loss_curves.png",
                     "Experiment 2a: Loss Curves (Distribution-Matched Random)")
    plot_cosine_histogram(all_results, exp_dir / "final_cosine_histogram.png",
                          "Experiment 2a: Final Cosine Similarity")
    save_metrics(all_results, exp_dir / "metrics.json")

    return all_results


def run_experiment2b(stats, n_targets=100, steps=DEFAULT_STEPS, n_restarts=DEFAULT_RESTARTS):
    """Experiment 2b: Raw random targets (norm-matched)."""
    print("\n" + "="*70)
    print("EXPERIMENT 2b: Raw Random Targets")
    print("="*70)

    exp_dir = RESULTS_DIR / "experiment2b"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Generate targets: random directions, scaled to match typical norms
    torch.manual_seed(123)
    targets = []
    for _ in range(n_targets):
        raw = torch.randn(SEQ_LEN, D_MODEL)
        # Scale each position to match typical real norm
        for pos in range(SEQ_LEN):
            raw[pos] = raw[pos] / raw[pos].norm() * stats["typical_norms"][pos]
        targets.append(raw.to(DEVICE))

    # Optimize
    all_results = []
    t0 = time.time()
    for i, target in enumerate(targets):
        print(f"\n2b Target {i+1}/{len(targets)}:")
        r = optimize_with_restarts(target, n_restarts=n_restarts, steps=steps)
        all_results.append(r)
        elapsed = time.time() - t0
        per_target = elapsed / (i + 1)
        remaining = per_target * (len(targets) - i - 1)
        print(f"  Best cosine: {r['best']['final_cosine']:.4f}, ETA: {remaining:.0f}s")

        if (i + 1) % 10 == 0:
            save_metrics(all_results, exp_dir / "metrics_checkpoint.json")

    # Generate outputs
    print("\nGenerating Experiment 2b outputs...")
    plot_loss_curves(all_results, exp_dir / "loss_curves.png",
                     "Experiment 2b: Loss Curves (Raw Random)")
    plot_cosine_histogram(all_results, exp_dir / "final_cosine_histogram.png",
                          "Experiment 2b: Final Cosine Similarity")
    save_metrics(all_results, exp_dir / "metrics.json")

    return all_results


# ── Experiment 3: Interpolation and Extrapolation ──────────────────────

def run_experiment3(targets, n_pairs=20, steps=DEFAULT_STEPS, n_restarts=DEFAULT_RESTARTS):
    """Experiment 3: probe geometry of reachable set via interpolation."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Interpolation and Extrapolation")
    print("="*70)

    exp_dir = RESULTS_DIR / "experiment3"
    exp_dir.mkdir(parents=True, exist_ok=True)

    alphas = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    # Select pairs
    np.random.seed(999)
    n_available = len(targets)
    pair_indices = [(np.random.randint(n_available), np.random.randint(n_available))
                    for _ in range(n_pairs)]
    # Ensure no self-pairs
    pair_indices = [(a, b) if a != b else (a, (b+1) % n_available)
                    for a, b in pair_indices]

    all_pair_results = []
    t0 = time.time()
    for pi, (idx_a, idx_b) in enumerate(pair_indices):
        A = targets[idx_a]
        B = targets[idx_b]
        pair_results = {}
        for alpha in alphas:
            interp_target = alpha * A + (1 - alpha) * B
            print(f"\nPair {pi+1}/{n_pairs}, alpha={alpha}:")
            r = optimize_with_restarts(interp_target, n_restarts=n_restarts, steps=steps)
            pair_results[alpha] = r
            print(f"  Best cosine: {r['best']['final_cosine']:.4f}")

        all_pair_results.append(pair_results)
        elapsed = time.time() - t0
        per_pair = elapsed / (pi + 1)
        remaining = per_pair * (n_pairs - pi - 1)
        print(f"  Pair elapsed: {elapsed:.0f}s, ETA: {remaining:.0f}s")

    # Plot reachability curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for pi, pair_results in enumerate(all_pair_results):
        cos_by_alpha = [pair_results[a]["best"]["final_cosine"] for a in alphas]
        ax.plot(alphas, cos_by_alpha, 'o-', alpha=0.3, color='blue', markersize=3)

    # Median curve
    median_by_alpha = []
    for a in alphas:
        vals = [pr[a]["best"]["final_cosine"] for pr in all_pair_results]
        median_by_alpha.append(np.median(vals))
    ax.plot(alphas, median_by_alpha, 'ro-', linewidth=2, markersize=8, label='Median')

    ax.set_xlabel('Alpha (interpolation parameter)')
    ax.set_ylabel('Final Cosine Similarity')
    ax.set_title('Experiment 3: Interpolation/Extrapolation Reachability')
    ax.axvspan(0, 1, alpha=0.1, color='green', label='Interpolation range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(exp_dir / "reachability_curve.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save metrics
    metrics = {
        "alphas": alphas,
        "n_pairs": n_pairs,
        "pair_indices": [(int(a), int(b)) for a, b in pair_indices],
        "median_cosine_by_alpha": {str(a): float(v) for a, v in zip(alphas, median_by_alpha)},
        "all_cosines_by_alpha": {
            str(a): [pr[a]["best"]["final_cosine"] for pr in all_pair_results]
            for a in alphas
        },
    }
    with open(exp_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    return all_pair_results


# ── Experiment 5: Target Corruption Sweep ──────────────────────────────

def run_experiment5(targets, stats, n_targets=50, steps=DEFAULT_STEPS, n_restarts=3):
    """Experiment 5: PCA direction replacement sweep."""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Target Corruption Sweep (PCA Variant)")
    print("="*70)

    exp_dir = RESULTS_DIR / "experiment5"
    exp_dir.mkdir(parents=True, exist_ok=True)

    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # PCA of all residuals
    all_res = stats["all_residuals"]  # [n_seq, SEQ_LEN, D_MODEL]
    # Reshape to [n_seq * SEQ_LEN, D_MODEL] for PCA
    flat = all_res.reshape(-1, D_MODEL)
    flat_centered = flat - flat.mean(dim=0)
    U, S, Vh = torch.linalg.svd(flat_centered, full_matrices=False)
    # Vh: [D_MODEL, D_MODEL] - PCA directions
    pca_dirs = Vh  # [D_MODEL, D_MODEL]

    # Empirical distribution of projections onto each PCA direction
    projections = flat_centered @ pca_dirs.T  # [n_samples, D_MODEL]
    proj_mean = projections.mean(dim=0)
    proj_std = projections.std(dim=0)

    use_targets = targets[:n_targets]

    all_frac_results = {f: [] for f in fractions}
    torch.manual_seed(777)
    t0 = time.time()

    for fi, frac in enumerate(fractions):
        print(f"\n--- Fraction {frac:.1f} ---")
        n_dirs = int(frac * D_MODEL)

        for ti, target in enumerate(use_targets):
            # Replace top n_dirs PCA directions with random values
            target_cpu = target.cpu()
            target_flat = target_cpu.reshape(-1, D_MODEL)  # [SEQ_LEN, D_MODEL]
            target_centered = target_flat - flat.mean(dim=0)

            if n_dirs > 0:
                # Project onto PCA directions
                proj = target_centered @ pca_dirs[:n_dirs].T  # [SEQ_LEN, n_dirs]
                # Replace with random values from empirical distribution
                new_proj = torch.randn_like(proj) * proj_std[:n_dirs] + proj_mean[:n_dirs]
                # Reconstruct
                diff = (new_proj - proj) @ pca_dirs[:n_dirs]
                corrupted = target_cpu + diff
            else:
                corrupted = target_cpu.clone()

            corrupted = corrupted.to(DEVICE)
            r = optimize_with_restarts(corrupted, n_restarts=n_restarts, steps=steps,
                                       log_every=0)
            all_frac_results[frac].append(r)

            if (ti + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  Target {ti+1}/{n_targets}, elapsed: {elapsed:.0f}s")

    # Plot: cosine sim vs fraction replaced
    fig, ax = plt.subplots(figsize=(10, 6))
    medians = []
    q25s = []
    q75s = []
    for frac in fractions:
        vals = [r["best"]["final_cosine"] for r in all_frac_results[frac]]
        medians.append(np.median(vals))
        q25s.append(np.percentile(vals, 25))
        q75s.append(np.percentile(vals, 75))

    ax.plot(fractions, medians, 'bo-', linewidth=2, markersize=8, label='Median')
    ax.fill_between(fractions, q25s, q75s, alpha=0.2, color='blue', label='IQR')
    ax.set_xlabel('Fraction of PCA Directions Replaced')
    ax.set_ylabel('Final Cosine Similarity')
    ax.set_title('Experiment 5: Target Corruption Sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(exp_dir / "corruption_curve.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save metrics
    metrics = {
        "fractions": fractions,
        "n_targets": n_targets,
        "median_cosine_by_fraction": {str(f): float(m) for f, m in zip(fractions, medians)},
        "all_cosines_by_fraction": {
            str(f): [r["best"]["final_cosine"] for r in all_frac_results[f]]
            for f in fractions
        },
    }
    with open(exp_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    return all_frac_results


# ── Summary Plot ───────────────────────────────────────────────────────

def generate_summary_plot(exp1_results, exp2a_results, exp2b_results,
                          exp3_results, exp4_results):
    """Generate the money plot comparing all experiments."""
    print("\nGenerating summary plot...")

    fig, ax = plt.subplots(figsize=(14, 7))

    data = {}
    labels = []

    # Exp 1: Real targets
    data["Real\nTargets"] = [r["best"]["final_cosine"] for r in exp1_results]
    labels.append("Real\nTargets")

    # Exp 2a: Distribution-matched
    if exp2a_results:
        data["Dist-Matched\nRandom"] = [r["best"]["final_cosine"] for r in exp2a_results]
        labels.append("Dist-Matched\nRandom")

    # Exp 2b: Raw random
    if exp2b_results:
        data["Raw\nRandom"] = [r["best"]["final_cosine"] for r in exp2b_results]
        labels.append("Raw\nRandom")

    # Exp 3: Interpolation alphas
    if exp3_results:
        alphas = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        for a in alphas:
            key = f"Interp\nα={a}"
            data[key] = [pr[a]["best"]["final_cosine"] for pr in exp3_results]
            labels.append(key)

    # Exp 4: Ablations
    if exp4_results:
        data["No FFN"] = [r["best"]["final_cosine"] for r in exp4_results["no_ffn"]]
        labels.append("No FFN")
        data["No Attn"] = [r["best"]["final_cosine"] for r in exp4_results["no_attn"]]
        labels.append("No Attn")

    positions = list(range(len(labels)))
    bp = ax.boxplot([data[l] for l in labels], positions=positions, widths=0.6,
                    patch_artist=True)

    # Color coding
    colors = []
    for l in labels:
        if "Real" in l:
            colors.append('#2196F3')
        elif "Dist" in l:
            colors.append('#4CAF50')
        elif "Raw" in l:
            colors.append('#FF9800')
        elif "Interp" in l:
            colors.append('#9C27B0')
        elif "FFN" in l or "Attn" in l:
            colors.append('#F44336')
        else:
            colors.append('#607D8B')

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Final Cosine Similarity')
    ax.set_title('Summary: Soft Prompt Reachability Across All Experiments')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {RESULTS_DIR / 'summary.png'}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    total_t0 = time.time()

    # ── Experiment 1 ──
    exp1_results, targets, real_all_residuals, sequences = run_experiment1(
        n_targets=100, steps=DEFAULT_STEPS, n_restarts=DEFAULT_RESTARTS
    )

    # Check if we need more steps
    median_cos = np.median([r["best"]["final_cosine"] for r in exp1_results])
    print(f"\nExperiment 1 median cosine: {median_cos:.4f}")

    # ── Experiment 1b ──
    exp1b_metrics = run_experiment1b(exp1_results)

    # ── Experiment 4 (ablation - early because it directly answers core question) ──
    exp4_results = run_experiment4(targets, steps=DEFAULT_STEPS,
                                   n_restarts=DEFAULT_RESTARTS,
                                   exp1_results=exp1_results)

    # ── Experiment 2a & 2b ──
    print("\nComputing activation statistics for Experiment 2...")
    stats = compute_activation_statistics(n_sequences=1000)

    exp2a_results = run_experiment2a(stats, n_targets=100, steps=DEFAULT_STEPS,
                                     n_restarts=DEFAULT_RESTARTS)
    exp2b_results = run_experiment2b(stats, n_targets=100, steps=DEFAULT_STEPS,
                                     n_restarts=DEFAULT_RESTARTS)

    # ── Experiment 3 ──
    exp3_results = run_experiment3(targets, n_pairs=20, steps=DEFAULT_STEPS,
                                   n_restarts=DEFAULT_RESTARTS)

    # ── Experiment 5 (if time permits) ──
    elapsed = time.time() - total_t0
    print(f"\nTotal elapsed: {elapsed:.0f}s")
    # Run exp5 if we have enough time remaining
    exp5_results = run_experiment5(targets, stats, n_targets=50,
                                   steps=DEFAULT_STEPS, n_restarts=3)

    # ── Summary ──
    generate_summary_plot(exp1_results, exp2a_results, exp2b_results,
                          exp3_results, exp4_results)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE. Total time: {total_elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
