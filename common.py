"""
Shared utilities for soft prompt reachability experiments.
"""

import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import json
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────

MODEL_NAME = "EleutherAI/pythia-160m"
TARGET_LAYER = 6
SEQ_LEN = 20
RESULTS_DIR = Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model Setup ────────────────────────────────────────────────────────

print(f"Loading model on {DEVICE}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()
for param in model.parameters():
    param.requires_grad = False
model = model.to(DEVICE)

D_MODEL = model.config.hidden_size
NUM_LAYERS = model.config.num_hidden_layers
print(f"Model ready: {NUM_LAYERS} layers, d_model={D_MODEL}")


# ── Core Functions ─────────────────────────────────────────────────────

def get_all_residuals(input_embeds):
    """Run input embeddings through model, return residual stream at all layers."""
    outputs = model(
        inputs_embeds=input_embeds.unsqueeze(0),
        output_hidden_states=True,
    )
    return [h.squeeze(0) for h in outputs.hidden_states]


def get_residual(input_embeds, layer=TARGET_LAYER):
    """Get residual at a specific layer."""
    return get_all_residuals(input_embeds)[layer]


def get_residual_fast(input_embeds, layer=TARGET_LAYER):
    """Get residual at a specific layer using partial forward (layers 0..layer-1 only).

    ~2x faster than get_residual() for layer=6 since it skips layers 6-11.
    """
    hidden = input_embeds.unsqueeze(0)
    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
    position_embeddings = model.gpt_neox.rotary_emb(hidden, position_ids=position_ids)
    for i in range(layer):
        hidden = model.gpt_neox.layers[i](
            hidden, position_embeddings=position_embeddings,
        )
    return hidden.squeeze(0)


def get_real_target(token_ids):
    """Get the layer-6 residual stream for a real token sequence."""
    with torch.no_grad():
        embeds = model.gpt_neox.embed_in(token_ids.unsqueeze(0)).squeeze(0)
        return get_residual(embeds).detach()


def get_real_all_residuals(token_ids):
    """Get all-layer residual streams for a real token sequence."""
    with torch.no_grad():
        embeds = model.gpt_neox.embed_in(token_ids.unsqueeze(0)).squeeze(0)
        return [r.detach() for r in get_all_residuals(embeds)]


def cosine_sim_per_position(achieved, target):
    """Per-position cosine similarity."""
    return F.cosine_similarity(achieved, target, dim=-1)


def optimize_prompt(target, steps=1000, lr=0.01, seed=None,
                    get_residual_fn=None, mse_weight=0.01, log_every=0):
    """Optimize a soft prompt to hit target residual stream activations."""
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
        cos_loss = 1 - cosine_sim_per_position(residual, target).mean()
        mse_loss = ((residual - target) ** 2).mean()
        loss = cos_loss + mse_weight * mse_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cos_val = 1 - cos_loss.item()
            mse_val = mse_loss.item()

        loss_history.append(loss.item())
        cosine_history.append(cos_val)
        mse_history.append(mse_val)

        if log_every and (step + 1) % log_every == 0:
            print(f"    Step {step+1:5d}: cos={cos_val:.4f}, mse={mse_val:.4f}")

    # Record all-layer residuals at final state
    with torch.no_grad():
        all_residuals = get_all_residuals(prompt.detach())

    return {
        "prompt": prompt.detach().cpu(),
        "loss_history": loss_history,
        "cosine_history": cosine_history,
        "mse_history": mse_history,
        "all_layer_residuals": [r.detach().cpu() for r in all_residuals],
        "final_cosine": cosine_history[-1],
        "final_mse": mse_history[-1],
    }


def optimize_with_restarts(target, n_restarts=3, **kwargs):
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


# ── Ablation Hooks ─────────────────────────────────────────────────────

def install_ablation_hooks(ablate="mlp"):
    """Install hooks to zero out MLP or attention outputs."""
    hooks = []
    for layer in model.gpt_neox.layers:
        if ablate == "mlp":
            target_module = layer.mlp
            def hook_fn(module, input, output):
                return torch.zeros_like(output)
        elif ablate == "attention":
            target_module = layer.attention
            def hook_fn(module, input, output):
                return (torch.zeros_like(output[0]),) + output[1:]
        else:
            raise ValueError(f"Unknown: {ablate}")

        h = target_module.register_forward_hook(hook_fn)
        hooks.append(h)
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ── Metrics & Plotting ─────────────────────────────────────────────────

def save_metrics(all_results, save_path, extra=None):
    """Save raw metrics to JSON."""
    finals_cos = [r["best"]["final_cosine"] for r in all_results]
    finals_mse = [r["best"]["final_mse"] for r in all_results]
    metrics = {
        "n_targets": len(all_results),
        "final_cosines": finals_cos,
        "final_mses": finals_mse,
        "restart_variances": [r["restart_variance"] for r in all_results],
        "all_restart_cosines": [r["all_final_cosines"] for r in all_results],
        "summary": {
            "mean_cosine": float(np.mean(finals_cos)),
            "median_cosine": float(np.median(finals_cos)),
            "std_cosine": float(np.std(finals_cos)),
            "min_cosine": float(np.min(finals_cos)),
            "max_cosine": float(np.max(finals_cos)),
            "mean_mse": float(np.mean(finals_mse)),
            "median_mse": float(np.median(finals_mse)),
        },
    }
    if extra:
        metrics.update(extra)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def plot_loss_curves(all_results, save_path, title="Loss Curves"):
    """Plot overlaid loss/cosine curves with median."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    cos_curves = [r["best"]["cosine_history"] for r in all_results]
    mse_curves = [r["best"]["mse_history"] for r in all_results]

    for c in cos_curves:
        ax1.plot(c, color='blue', alpha=0.1, linewidth=0.5)
    for m in mse_curves:
        ax2.plot(m, color='red', alpha=0.1, linewidth=0.5)

    cos_arr = np.array(cos_curves)
    mse_arr = np.array(mse_curves)
    ax1.plot(np.median(cos_arr, axis=0), color='blue', linewidth=2, label='Median Cosine Sim')
    ax2.plot(np.median(mse_arr, axis=0), color='red', linewidth=2, label='Median MSE')

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cosine Similarity', color='blue')
    ax2.set_ylabel('MSE', color='red')
    ax1.set_title(title)
    ax1.legend(loc='center left')
    ax2.legend(loc='center right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cosine_histogram(all_results, save_path, title="Final Cosine Similarity"):
    """Histogram of best-of-restarts final cosine similarity."""
    finals = [r["best"]["final_cosine"] for r in all_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(finals, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Final Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.axvline(np.median(finals), color='red', linestyle='--',
               label=f'Median: {np.median(finals):.4f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cosine_by_position(all_results, targets, save_path, title="Cosine by Position",
                            get_residual_fn=None):
    """Per-position cosine similarity with error bars."""
    if get_residual_fn is None:
        get_residual_fn = lambda x: get_residual(x)
    per_pos = []
    for r, t in zip(all_results, targets):
        prompt = r["best"]["prompt"].to(DEVICE)
        with torch.no_grad():
            achieved = get_residual_fn(prompt)
            cos = cosine_sim_per_position(achieved, t).cpu().numpy()
        per_pos.append(cos)
    per_pos = np.array(per_pos)
    means = per_pos.mean(axis=0)
    stds = per_pos.std(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(range(SEQ_LEN), means, yerr=stds, fmt='o-', capsize=3)
    ax.set_xlabel('Position')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(title)
    ax.set_xticks(range(SEQ_LEN))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_layer_trajectories(results, targets, real_all_residuals_list, save_path,
                            title="Layer Trajectories"):
    """Plot per-layer cosine sim for best/median/worst cases."""
    finals = [r["best"]["final_cosine"] for r in results]
    sorted_idx = np.argsort(finals)
    cases = {
        "Best": sorted_idx[-1],
        "Median": sorted_idx[len(sorted_idx) // 2],
        "Worst": sorted_idx[0],
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (label, idx) in zip(axes, cases.items()):
        opt_res = results[idx]["best"]["all_layer_residuals"]
        real_res = real_all_residuals_list[idx]
        target = targets[idx]
        opt_cos, real_cos = [], []
        for li in range(len(opt_res)):
            opt_r = opt_res[li].to(DEVICE)
            real_r = real_res[li].to(DEVICE)
            opt_cos.append(cosine_sim_per_position(opt_r, target).mean().item())
            real_cos.append(cosine_sim_per_position(real_r, target).mean().item())
        layers = list(range(len(opt_res)))
        ax.plot(layers, opt_cos, 'b-o', label='Optimized', markersize=4)
        ax.plot(layers, real_cos, 'r-s', label='Real', markersize=4)
        ax.axvline(TARGET_LAYER, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Layer')
        ax.set_title(f'{label} (cos={finals[idx]:.4f})')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Cosine Sim to Target')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
