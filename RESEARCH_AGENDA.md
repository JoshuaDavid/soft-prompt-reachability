# Soft Prompt Reachability of Residual Stream Targets

## Motivation

Mechanistic interpretability has gotten good at identifying *what* internal
representations correspond to specific model behaviors. Activation patching,
steering vectors, and sparse autoencoders can locate directions in residual
space that, when amplified or suppressed, predictably change model outputs.

But there's a gap between "I know which residual-stream direction produces the
behavior I want" and "I can actually make the model enter that state." The
standard tools for influencing a model at inference time are constrained:
prompt engineering operates through discrete tokens, soft prompt tuning gives
you continuous control but only at layer 0, and activation patching requires
runtime access to model internals (which you typically don't have in
deployment).

This raises a concrete question: **if you identify a target activation pattern
at some intermediate layer, can you find a layer-0 input that produces it?**

If the answer is "yes, almost always," that has implications in several
directions:

- **Soft prompt attacks.** If arbitrary mid-layer states are reachable from
  layer-0 inputs, then any behavior achievable via activation patching is also
  achievable via a soft prompt. This means the attack surface of soft-prompt-
  capable APIs is at least as large as the set of behaviors discoverable via
  interpretability tools.

- **Prompt-based steering.** Conversely, if you want to *beneficially* steer a
  model (e.g., elicit a specific reasoning pattern), reachability tells you
  whether prompt optimization is a viable path or whether you need internal
  access.

- **Understanding the geometry of transformer computation.** The reachable set
  from layer-0 inputs characterizes what the first L layers can express as a
  function. If this set is nearly all of R^d, the layers are very expressive
  and the residual stream is doing something closer to arbitrary computation
  than to a constrained pipeline. If the reachable set is thin, then the
  layers are imposing strong inductive biases that limit what states are
  possible, regardless of the input.

- **Relating the FFN nonlinearity to expressivity.** Theoretically, the ReLU/
  GeLU in each FFN layer should expand the reachable set (more linear regions
  in the composed function). But does this matter in practice, or does the
  residual connection dominate so thoroughly that the nonlinearity is
  irrelevant to reachability? Ablation experiments (comparing reachability
  with and without FFNs) can answer this directly.

We start with the easy version of the problem ⸻ unconstrained vectors in
R^d_model, not actual soft prompts ⸻ because if even this relaxed problem
shows unreachability, the harder constrained version is dead on arrival. If the
easy version works, we can then check whether the solutions happen to land near
the token embedding manifold (in which case the constrained version may be
equally easy) or far from it (in which case the gap between "theoretically
reachable" and "reachable via real tokens" becomes the interesting question).

## Research Question

Given a length-20 input in R^(20 × d_model), optimized via gradient descent,
what fraction of target residual stream activations at a middle layer are
reachable? How does reachability depend on the FFN nonlinearity vs. attention
mixing? And how far are the optimized inputs from the token embedding manifold?

## Theoretical Prior

Most targets should be reachable, for three reasons:

1. **Residual connections preserve dimensionality.** Each layer computes
   `x_{l+1} = x_l + Attn(x_l) + FFN(x_l)`. The map is a perturbation of
   identity, hence a diffeomorphism when corrections are moderate. This
   prevents collapse into low-dimensional subspaces.

2. **Nonlinearities add expressivity.** Without them, 12 residual layers
   compose to `x + Ax` for some matrix A ⸻ a single affine map. ReLU/GeLU
   creates folding across activation regimes, making the composed reachable
   set much larger than any affine subspace.

3. **Degrees of freedom roughly match.** A length-20 input in d_model=768
   gives 15,360 free parameters. Targeting 20 positions × 768 dimensions =
   15,360 target values. The attention mechanism couples positions, so it's
   not 20 independent problems, but the dimensionality is in the right
   ballpark.

**Expected failure modes:**
- Targets requiring neurons to be in contradictory activation regimes across
  positions that attend to each other.
- Targets far from the natural manifold, requiring corrections outside the
  conic hull of W_down columns at critical layers.
- Optimization failures (bad local minima) rather than true unreachability.

## Model

**Pythia-160M** (`EleutherAI/pythia-160m`)
- 12 layers, hidden_size=768, 162M parameters
- Small enough to iterate fast, big enough to be a real transformer
- Embedding layer: `model.gpt_neox.embed_in`
- Target layer: 6 (middle of the network)
- Requires `output_hidden_states=True` to extract intermediate residuals

## Metrics (All Experiments)

**Primary metric: per-position cosine similarity** between achieved and target
residual vectors, averaged across positions.

LayerNorm renormalizes the residual stream before each sublayer, so the
*direction* of the residual vector matters more than its magnitude for
downstream computation. Cosine similarity captures this.

**Secondary metric: MSE.** Still worth reporting because scale differences
can flip ReLU activations in subsequent layers, so a vector that is
cosine-similar but wrong-scale may still land in a different computational
basin. Reporting both is cheap.

**Diagnostic metrics (recorded for all experiments):**

- Loss curves (primary + secondary) over optimization steps
- Per-position cosine similarity breakdown (are early positions easier?)
- All-layer residual trajectories: record residual stream at every layer
  (0 through 12), not just the target layer. This reveals *where* optimization
  gets stuck when it fails, and whether successful optimizations take
  "natural" paths through intermediate layers or do something deranged.
- Restart variance: spread of final loss across random restarts for each
  target (distinguishes "hard to optimize" from "unreachable")

## Experiment 1: Reachability of Real Activations

**Purpose:** Calibration baseline. These targets are known-reachable (some
input produces them). Tests whether gradient descent can find an input that
reproduces them.

**Procedure:**
1. Sample 100 real sequences from `wikitext-2`, truncated/padded to 20 tokens.
2. For each sequence, run it through the model and record the layer-6
   residual stream as the target (shape: `[20, 768]`).
3. Initialize a random input `P ~ N(0, 1)` of shape `[20, 768]`.
4. Optimize P via Adam to minimize a combined loss:
   `loss = (1 - cosine_sim).mean() + λ * mse` with λ=0.01 (or tune).
5. Run 5 random restarts per target.

**Hyperparameters:**
- Optimizer: Adam, lr=0.01
- Steps: 2000 (extend to 5000 if loss curves haven't plateaued)
- 5 random restarts per target

### Experiment 1b: Embedding Manifold Distance

**Purpose:** Check whether the unconstrained optimization (arbitrary R^768
vectors) finds solutions near real token embeddings, or exploits regions of
embedding space that no token occupies.

**Procedure:** After Experiment 1 converges, for each of the 20 optimized
input vectors in each successful result:

1. Compute the nearest neighbor in the token embedding matrix
   (`model.gpt_neox.embed_in.weight`) by cosine similarity.
2. Compute the distance to the convex hull of token embeddings (or,
   as a cheaper proxy, the distance to the affine subspace spanned by the
   top-k PCA components of the embedding matrix).
3. Report: distribution of nearest-neighbor cosine similarities,
   distribution of distances, and whether the optimized vectors cluster
   near real token embeddings or exploit regions of R^768 that no token
   occupies.

**Interpretation:** If optimized vectors are close to real embeddings,
the unconstrained-vs-constrained distinction is moot and the results
generalize to real soft prompts. If they're far away, the constrained
version is a meaningful follow-up.

This is ~5 lines of code on top of Experiment 1 results.

## Experiment 2: Random Targets

**Purpose:** Test reachability of targets not known to correspond to any real
input. Two sub-experiments probe different distances from the natural manifold.

### Experiment 2a: Distribution-Matched Random Targets

1. Compute per-position mean μ_i and covariance Σ_i of layer-6 residuals
   across 1000+ real sequences from wikitext-2. (Per-position because
   positional structure matters.)
2. Sample 100 targets where position i is drawn from N(μ_i, Σ_i).
3. Optimize as in Experiment 1.

### Experiment 2b: Raw Random Targets

1. Compute the typical per-position L2 norm of real layer-6 residuals.
2. Sample 100 targets as `randn(20, 768)`, scaled so each position's norm
   matches the typical real norm.
3. Optimize as above.

**Expected:** 2a should be substantially easier than 2b. The gap tells you how
much of residual space is "wasted" on unreachable regions vs. how much the
optimization landscape matters.

## Experiment 3: Interpolation and Extrapolation

**Purpose:** Probe the geometry of the reachable set. If the reachable set is
convex, interpolations should be easy. Extrapolations test the boundary.

**Procedure:**
1. Select 20 pairs of real activation patterns (A, B) from Experiment 1.
2. For each pair, set target = αA + (1-α)B for α ∈ {-0.5, 0, 0.25, 0.5,
   0.75, 1.0, 1.5, 2.0}.
3. Optimize as in Experiment 1.

**Metrics:**
- Final cosine similarity and MSE as a function of α
- Plot the "reachability curve" for each pair

**Expected:** α ∈ [0,1] should be easy. α outside [0,1] should degrade, with
the degradation rate indicating how far the reachable set extends beyond the
convex hull of real activations.

## Experiment 4: FFN and Attention Ablation

**Purpose:** Directly answer the original question ⸻ are FFN nonlinearities
net-positive or net-negative for reachability? ⸻ by removing them and
measuring the effect. Also isolates the attention contribution.

**Procedure:** Re-run Experiment 1 (same 100 targets, same optimization
setup) under three model variants:

### 4a: FFN Ablated (Attention Only)

Hook every layer to zero out the FFN contribution, so each layer computes:
`x_{l+1} = x_l + Attn(x_l)`

This removes all nonlinearities. The composed map is now a sequence of
residual linear-ish operations (attention is still data-dependent but
piecewise-linear in the input, minus the softmax nonlinearity). If
reachability drops, FFNs are net-positive. If it improves or stays flat,
FFNs aren't load-bearing for reachability.

### 4b: Attention Ablated (FFN Only)

Hook every layer to zero out the attention contribution:
`x_{l+1} = x_l + FFN(x_l)`

Now positions are completely decoupled. Each position is an independent
12-layer MLP with residual connections. This should make per-position
reachability *easier* (no cross-position coupling) but removes the model's
ability to mix information across positions. Compare per-position cosine
similarity with the full model.

### 4c: Full Model (Control)

Same as Experiment 1, repeated for consistency. (Can reuse Experiment 1
results if seeds match.)

**Metrics:** Same as Experiment 1, plus:
- Pairwise comparison of final cosine similarity: full vs. FFN-ablated,
  full vs. attention-ablated
- Per-target Δ cosine similarity (does the same target that's hard for the
  full model also become hard under ablation, or do they fail differently?)

**Note on hooks:** The ablation hooks need to intercept specific sublayer
outputs. Inspect the model architecture with `print(model)` to identify
exact module paths. For Pythia/GPTNeoX, the relevant modules are likely:
- `model.gpt_neox.layers[i].attention` (attention output)
- `model.gpt_neox.layers[i].mlp` (FFN output)

Register forward hooks that zero out the relevant sublayer's contribution
to the residual stream.

**Important caveat:** The targets in Experiment 1 were generated by the
full model. When you ablate the FFN or attention, the layer-6 residual
for the *original* token sequence will be different from the Experiment 1
target. The question being asked is: "can the ablated model reach the
same targets as the full model?" not "can the ablated model reproduce its
own outputs?" This is intentional ⸻ it tests whether the FFN/attention
is necessary for reaching those specific regions of activation space.

## Experiment 5: Target Corruption Sweep (Optional / Exploratory)

**Purpose:** Find the transition boundary between reachable and unreachable by
gradually corrupting a known-reachable target.

**Note:** This experiment operates on raw dimensions, which are not
individually meaningful ⸻ features live in superposition across dimensions.
Results characterize robustness of reachability to target perturbation, not
individual feature controllability. A cleaner version would replace PCA
directions of real activation data rather than raw dimensions; both variants
are described below.

### Variant A: Dimension Replacement

1. Take 50 real activation targets from Experiment 1.
2. For each target, for fraction f ∈ {0, 0.1, 0.2, ..., 0.9, 1.0}:
   a. Randomly select f × 768 dimensions (same dimensions across all 20
      positions, to keep it clean).
   b. Replace those dimensions with random values, sampled from N(0, σ)
      where σ is the empirical std of that dimension across real data.
   c. Optimize as in Experiment 1.
3. Run 3 random restarts per (target, fraction) pair.

### Variant B: PCA Direction Replacement (Preferred if Setup Cost is Low)

1. Compute PCA of layer-6 residuals across 1000+ real sequences.
2. Take 50 real activation targets.
3. For each target, for fraction f ∈ {0, 0.1, 0.2, ..., 0.9, 1.0}:
   a. Select the top f × 768 PCA directions.
   b. Replace the target's projection onto those directions with random
      values drawn from the empirical distribution of projections onto
      those directions.
   c. Optimize as in Experiment 1.

Variant B replaces the most structured (highest-variance) directions first,
which is a more meaningful corruption than replacing arbitrary dimensions.

**Metrics:**
- Final cosine similarity vs. fraction replaced
- Look for: gradual degradation (reachable set is "thick") vs. sharp
  transition (reachable set has a clear boundary)

## Implementation

```python
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Setup ──────────────────────────────────────────────────────────────

model_name = "EleutherAI/pythia-160m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# Freeze all model parameters
for param in model.parameters():
    param.requires_grad = False

TARGET_LAYER = model.config.num_hidden_layers // 2  # middle layer (~6)
SEQ_LEN = 20
D_MODEL = model.config.hidden_size  # should be 768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# ── Core Functions ─────────────────────────────────────────────────────

def get_all_residuals(input_embeds):
    """
    Run input embeddings through model, return residual stream at
    ALL layers.

    Args:
        input_embeds: Tensor of shape [SEQ_LEN, D_MODEL]
    Returns:
        List of Tensors, each [SEQ_LEN, D_MODEL], from layer 0 to layer 12
    """
    outputs = model(
        inputs_embeds=input_embeds.unsqueeze(0),
        output_hidden_states=True,
    )
    return [h.squeeze(0) for h in outputs.hidden_states]


def get_residual(input_embeds, layer=TARGET_LAYER):
    """Convenience wrapper: single layer's residual."""
    return get_all_residuals(input_embeds)[layer]


def get_real_target(token_ids):
    """
    Get the residual stream for a real token sequence.

    Args:
        token_ids: Tensor of shape [SEQ_LEN] (long)
    Returns:
        Tensor of shape [SEQ_LEN, D_MODEL], detached
    """
    embeds = model.gpt_neox.embed_in(token_ids.unsqueeze(0)).squeeze(0)
    return get_residual(embeds).detach()


def cosine_loss(achieved, target):
    """
    1 - mean per-position cosine similarity.
    Returns a scalar loss in [0, 2].
    """
    cos = torch.nn.functional.cosine_similarity(achieved, target, dim=-1)
    return 1 - cos.mean()


def combined_loss(achieved, target, mse_weight=0.01):
    """Primary: cosine similarity. Secondary: MSE (small weight)."""
    cos = cosine_loss(achieved, target)
    mse = ((achieved - target) ** 2).mean()
    return cos + mse_weight * mse


def optimize_prompt(target, steps=2000, lr=0.01, seed=None,
                    get_residual_fn=None):
    """
    Optimize a soft prompt to hit target residual stream activations.

    Args:
        target: Tensor of shape [SEQ_LEN, D_MODEL]
        steps: Number of optimization steps
        lr: Learning rate
        seed: Random seed for reproducibility
        get_residual_fn: Optional custom residual function (for ablations).
                         Defaults to get_residual.

    Returns:
        dict with keys:
            prompt: optimized input [SEQ_LEN, D_MODEL]
            loss_history: list of float
            cosine_history: list of float
            mse_history: list of float
            all_layer_residuals: list of [SEQ_LEN, D_MODEL] at final step
    """
    if get_residual_fn is None:
        get_residual_fn = get_residual

    if seed is not None:
        torch.manual_seed(seed)

    prompt = torch.randn(SEQ_LEN, D_MODEL, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([prompt], lr=lr)

    loss_history, cosine_history, mse_history = [], [], []
    for step in range(steps):
        optimizer.zero_grad()
        residual = get_residual_fn(prompt)
        loss = combined_loss(residual, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cos = torch.nn.functional.cosine_similarity(
                residual, target, dim=-1
            ).mean().item()
            mse = ((residual - target) ** 2).mean().item()

        loss_history.append(loss.item())
        cosine_history.append(cos)
        mse_history.append(mse)

    # Record all-layer residuals at final state
    with torch.no_grad():
        all_residuals = get_all_residuals(prompt.detach())

    return {
        "prompt": prompt.detach(),
        "loss_history": loss_history,
        "cosine_history": cosine_history,
        "mse_history": mse_history,
        "all_layer_residuals": [r.detach() for r in all_residuals],
    }


def optimize_with_restarts(target, n_restarts=5, **kwargs):
    """
    Run multiple random restarts and return the best result.

    Returns:
        dict with keys:
            best: result dict from best restart
            all_final_losses: list of final losses from each restart
    """
    best_result = None
    best_final = float("inf")
    all_final = []

    for i in range(n_restarts):
        result = optimize_prompt(target, seed=i * 1000, **kwargs)
        final = result["loss_history"][-1]
        all_final.append(final)
        if final < best_final:
            best_final = final
            best_result = result

    return {"best": best_result, "all_final_losses": all_final}


# ── Embedding Manifold Distance (Experiment 1b) ───────────────────────

def embedding_manifold_distances(optimized_prompts):
    """
    For each optimized input vector, compute:
    - cosine similarity to nearest token embedding
    - index of nearest token embedding
    - L2 distance to nearest token embedding

    Args:
        optimized_prompts: list of Tensors [SEQ_LEN, D_MODEL]
    Returns:
        dict of lists
    """
    embed_matrix = model.gpt_neox.embed_in.weight.detach()  # [vocab, D_MODEL]
    embed_normed = embed_matrix / embed_matrix.norm(dim=-1, keepdim=True)

    all_cos_sims = []
    all_l2_dists = []
    all_nearest_tokens = []

    for prompt in optimized_prompts:
        prompt_normed = prompt / prompt.norm(dim=-1, keepdim=True)
        # [SEQ_LEN, vocab]
        cos_sims = prompt_normed @ embed_normed.T
        best_cos, best_idx = cos_sims.max(dim=-1)  # [SEQ_LEN]
        best_embeds = embed_matrix[best_idx]  # [SEQ_LEN, D_MODEL]
        l2_dists = (prompt - best_embeds).norm(dim=-1)  # [SEQ_LEN]

        all_cos_sims.append(best_cos.cpu().numpy())
        all_l2_dists.append(l2_dists.cpu().numpy())
        all_nearest_tokens.append(best_idx.cpu().numpy())

    return {
        "cosine_to_nearest": all_cos_sims,
        "l2_to_nearest": all_l2_dists,
        "nearest_token_ids": all_nearest_tokens,
    }


# ── Ablation Hooks (Experiment 4) ─────────────────────────────────────

# NOTE: Inspect `print(model)` to confirm exact module paths before using.
# For Pythia/GPTNeoX, the relevant modules are:
#   model.gpt_neox.layers[i].attention
#   model.gpt_neox.layers[i].mlp
#
# Strategy: register forward hooks on the mlp or attention submodules
# that replace their output with zeros. Since the layer adds the sublayer
# output to the residual, zeroing the sublayer output effectively removes
# its contribution.
#
# Example (sketch ⸻ test and adjust based on actual forward signature):
#
#   hooks = []
#   for layer in model.gpt_neox.layers:
#       h = layer.mlp.register_forward_hook(
#           lambda mod, inp, out: torch.zeros_like(out)
#       )
#       hooks.append(h)
#
#   # ... run optimization ...
#
#   for h in hooks:
#       h.remove()


# ── Data Loading ───────────────────────────────────────────────────────

def load_real_sequences(n=100):
    """
    Load n real token sequences of length SEQ_LEN from wikitext-2.
    Returns list of tensors, each of shape [SEQ_LEN].
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    sequences = []
    for example in dataset:
        text = example["text"].strip()
        if len(text) < 20:
            continue
        tokens = tokenizer.encode(text, return_tensors="pt").squeeze(0)
        if tokens.shape[0] >= SEQ_LEN:
            sequences.append(tokens[:SEQ_LEN].to(device))
        if len(sequences) >= n:
            break
    return sequences
```

## Outputs

For each experiment, produce:

1. **`results/{experiment_name}/loss_curves.png`** ⸻ All loss curves
   overlaid, with median highlighted. Plot both cosine similarity and MSE
   on dual y-axes.

2. **`results/{experiment_name}/final_cosine_histogram.png`** ⸻ Histogram of
   best-of-restarts final cosine similarity. Look for bimodality (some
   reachable, some not) vs. unimodal near 1.0 (mostly reachable).

3. **`results/{experiment_name}/cosine_by_position.png`** ⸻ Per-position
   cosine similarity, averaged across targets with error bars. Are early
   positions easier to control?

4. **`results/{experiment_name}/layer_trajectories.png`** ⸻ For a handful
   of representative targets (best case, worst case, median), plot
   per-layer cosine similarity between the optimized input's trajectory
   and the real input's trajectory through layers 0-12. Divergence at
   early layers = attention coupling problem. Divergence at late layers
   = FFN expressivity problem.

5. **`results/{experiment_name}/metrics.json`** ⸻ Raw numbers: per-target
   final cosine sim, final MSE, per-position cosine similarity, restart
   variance, all-layer cosine similarities.

6. **`results/experiment1b_embedding_distance.png`** ⸻ Histogram of
   cosine similarity between optimized input vectors and their nearest
   token embedding. If this clusters near 1.0, the unconstrained problem
   is a good proxy for the constrained (real soft prompt) problem.

7. **`results/summary.png`** ⸻ The money plot. X-axis: experiment condition
   (real targets, distribution-matched random, raw random, interpolation
   alphas). Y-axis: final cosine similarity. Box plots or violin plots.

8. **`results/ablation_comparison.png`** ⸻ Paired comparison: for each of
   the 100 targets, plot full-model cosine sim vs. FFN-ablated cosine sim
   (one subplot) and full-model vs. attention-ablated (another subplot).
   Points above the diagonal = ablation helped; below = ablation hurt.

## Interpretation Guide

| Observation | Interpretation |
|---|---|
| Exp 1 all converge to cosine ~1.0 | Optimization works; baseline is clean |
| Exp 1 some don't converge | Optimization has bad local minima; increase restarts/steps before drawing conclusions |
| Exp 1b embeddings near manifold | Unconstrained results generalize to real soft prompts |
| Exp 1b embeddings far from manifold | Need a constrained follow-up to say anything about real soft prompts |
| Exp 2a ≈ Exp 1 | Distribution-matched random targets are just as reachable as real ones |
| Exp 2b much worse than 2a | The reachable set is concentrated near the natural manifold |
| Exp 2b ≈ Exp 2a | Most of residual space is reachable; nonlinearities aren't the bottleneck |
| Exp 3 smooth across α ∈ [0,1] | Reachable set is approximately convex near the data manifold |
| Exp 3 sharp degradation at α > 1 | Reachable set doesn't extend far beyond the convex hull of real activations |
| Exp 4a (no FFN) much worse than 4c | FFN nonlinearities are net-positive for reachability |
| Exp 4a ≈ 4c | FFNs aren't load-bearing; residual connections + attention suffice |
| Exp 4b (no attn) per-position better, cross-position worse | Attention coupling is a double-edged sword: enables cross-position control but creates interference |
| Exp 4b uniformly better than 4c | Attention coupling is the primary obstacle to reachability |

## Execution Order

Run in this order, since later experiments depend on insights from earlier ones:

1. **Experiment 1** first (calibration; generates real targets reused later).
   Then immediately run **Experiment 1b** on the results.
2. **Experiment 4** (ablation). Run this early because it directly answers the
   original question and is cheap (same targets, same setup, just hooked
   model). Requires getting the hooks right, so inspect `print(model)` first.
3. **Experiment 2a** then **2b** (may want to adjust step count based on Exp 1).
4. **Experiment 3** (uses pairs from Exp 1).
5. **Experiment 5** (optional; run if time permits and earlier results raise
   questions about the boundary of the reachable set).

If Experiment 1 shows that 2000 steps isn't enough (loss curves still
declining at step 2000), increase to 5000 everywhere before proceeding.
