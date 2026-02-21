# Soft Prompt Reachability of Residual Stream Targets

## Research Question

Given a length-20 soft prompt optimized via gradient descent, what fraction of
target residual stream activations at the final layer are reachable? Are the
FFN nonlinearities net-positive or net-negative for reachability, and where is
the boundary of the reachable set relative to the natural data manifold?

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

3. **Degrees of freedom roughly match.** A length-20 soft prompt in
   d_model=768 gives 15,360 free parameters. Targeting 20 positions × 768
   dimensions = 15,360 target values. The attention mechanism couples
   positions, so it's not 20 independent problems, but the dimensionality
   is in the right ballpark.

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
- Target layer: 12 (final)
- Requires `output_hidden_states=True` to extract intermediate residuals

## Experiment 1: Reachability of Real Activations

**Purpose:** Calibration baseline. These targets are known-reachable (some
input produces them). Tests whether gradient descent can find a soft prompt
that reproduces them.

**Procedure:**
1. Sample 100 real sequences from `wikitext-2`, truncated/padded to 20 tokens.
2. For each sequence, run it through the model and record the layer-12
   residual stream as the target (shape: `[20, 768]`).
3. Initialize a random soft prompt `P ~ N(0, 1)` of shape `[20, 768]`.
4. Optimize P via Adam to minimize MSE between `get_residual(P)` and target.
5. Run 5 random restarts per target.

**Hyperparameters:**
- Optimizer: Adam, lr=0.01
- Steps: 2000 (extend to 5000 if loss curves haven't plateaued)
- 5 random restarts per target

**Metrics:**
- Final MSE (best of 5 restarts)
- Per-position cosine similarity
- Loss curve shape (converging to zero vs. plateau)
- Fraction of targets where best-of-5 MSE < threshold (define threshold from
  the distribution; e.g., 1% of initial MSE)

## Experiment 2: Random Targets

**Purpose:** Test reachability of targets not known to correspond to any real
input. Two sub-experiments probe different distances from the natural manifold.

### Experiment 2a: Distribution-Matched Random Targets

1. Compute per-position mean μ_i and covariance Σ_i of layer-12 residuals
   across 1000+ real sequences from wikitext-2. (Per-position because
   positional structure matters.)
2. Sample 100 targets where position i is drawn from N(μ_i, Σ_i).
3. Optimize soft prompts as in Experiment 1.

### Experiment 2b: Raw Random Targets

1. Compute the typical per-position L2 norm of real layer-12 residuals.
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
3. Optimize soft prompts as in Experiment 1.

**Metrics:**
- Final MSE as a function of α
- Plot the "reachability curve" for each pair

**Expected:** α ∈ [0,1] should be easy. α outside [0,1] should degrade, with
the degradation rate indicating how far the reachable set extends beyond the
convex hull of real activations.

## Experiment 4: Dimension Replacement Sweep

**Purpose:** Find the transition boundary between reachable and unreachable by
gradually corrupting a known-reachable target.

**Procedure:**
1. Take 50 real activation targets from Experiment 1.
2. For each target, for fraction f ∈ {0, 0.1, 0.2, ..., 0.9, 1.0}:
   a. Randomly select f × 768 dimensions (same dimensions across all 20
      positions, to keep it clean).
   b. Replace those dimensions with random values, sampled from N(0, σ)
      where σ is the empirical std of that dimension across real data.
   c. Optimize a soft prompt to hit this partially-corrupted target.
3. Run 3 random restarts per (target, fraction) pair.

**Metrics:**
- Final MSE vs. fraction replaced
- Look for: gradual degradation (reachable set is "thick") vs. sharp
  transition (reachable set has a clear boundary)

**This is the most informative experiment.** A sharp transition tells you the
reachable set is thin in some directions. Gradual degradation suggests most
of residual space is reachable and failures are optimization-related.

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

TARGET_LAYER = model.config.num_hidden_layers  # should be 12
SEQ_LEN = 20
D_MODEL = model.config.hidden_size  # should be 768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# ── Core Functions ─────────────────────────────────────────────────────

def get_residual(input_embeds, layer=TARGET_LAYER):
    """
    Run input embeddings through model, return residual stream at
    target layer.

    Args:
        input_embeds: Tensor of shape [SEQ_LEN, D_MODEL]
    Returns:
        Tensor of shape [SEQ_LEN, D_MODEL]
    """
    outputs = model(
        inputs_embeds=input_embeds.unsqueeze(0),
        output_hidden_states=True,
    )
    return outputs.hidden_states[layer].squeeze(0)


def get_real_target(token_ids):
    """
    Get the layer-12 residual stream for a real token sequence.

    Args:
        token_ids: Tensor of shape [SEQ_LEN] (long)
    Returns:
        Tensor of shape [SEQ_LEN, D_MODEL], detached
    """
    embeds = model.gpt_neox.embed_in(token_ids.unsqueeze(0)).squeeze(0)
    return get_residual(embeds).detach()


def optimize_prompt(target, steps=2000, lr=0.01, seed=None):
    """
    Optimize a soft prompt to hit target residual stream activations.

    Args:
        target: Tensor of shape [SEQ_LEN, D_MODEL]
        steps: Number of optimization steps
        lr: Learning rate
        seed: Random seed for reproducibility

    Returns:
        (optimized_prompt, loss_history)
    """
    if seed is not None:
        torch.manual_seed(seed)

    prompt = torch.randn(SEQ_LEN, D_MODEL, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([prompt], lr=lr)

    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        residual = get_residual(prompt)
        loss = ((residual - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return prompt.detach(), losses


def optimize_with_restarts(target, n_restarts=5, **kwargs):
    """
    Run multiple random restarts and return the best result.

    Returns:
        (best_prompt, best_losses, all_final_losses)
    """
    best_prompt, best_losses = None, None
    best_final = float("inf")
    all_final = []

    for i in range(n_restarts):
        prompt, losses = optimize_prompt(target, seed=i * 1000, **kwargs)
        final = losses[-1]
        all_final.append(final)
        if final < best_final:
            best_final = final
            best_prompt = prompt
            best_losses = losses

    return best_prompt, best_losses, all_final


# ── Metrics ────────────────────────────────────────────────────────────

def per_position_cosine_similarity(achieved, target):
    """Cosine similarity at each of the 20 positions."""
    # achieved, target: [SEQ_LEN, D_MODEL]
    cos = torch.nn.functional.cosine_similarity(achieved, target, dim=-1)
    return cos  # [SEQ_LEN]


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
   overlaid, with median highlighted.

2. **`results/{experiment_name}/final_mse_histogram.png`** ⸻ Histogram of
   best-of-restarts final MSE. Look for bimodality (some reachable, some not)
   vs. unimodal near zero (mostly reachable).

3. **`results/{experiment_name}/cosine_by_position.png`** ⸻ Per-position
   cosine similarity, averaged across targets with error bars. Are early
   positions easier to control?

4. **`results/{experiment_name}/metrics.json`** ⸻ Raw numbers: per-target
   final MSE, per-position cosine similarity, restart variance.

5. **`results/summary.png`** ⸻ The money plot. X-axis: experiment condition
   (real targets, distribution-matched random, raw random, interpolation
   alphas, dimension-replacement fractions). Y-axis: final MSE (log scale).
   Box plots or violin plots.

6. **`results/experiment4_transition.png`** ⸻ The most informative single
   plot. X: fraction of dimensions replaced. Y: final MSE. One line per
   target (light), median (bold). Sharp elbow = thin reachable set. Gradual
   slope = thick reachable set.

## Interpretation Guide

| Observation | Interpretation |
|---|---|
| Exp 1 all converge to ~0 | Optimization works; baseline is clean |
| Exp 1 some don't converge | Optimization has bad local minima; increase restarts/steps before drawing conclusions |
| Exp 2a ≈ Exp 1 | Distribution-matched random targets are just as reachable as real ones |
| Exp 2b much worse than 2a | The reachable set is concentrated near the natural manifold |
| Exp 2b ≈ Exp 2a | Most of residual space is reachable; nonlinearities aren't the bottleneck |
| Exp 3 smooth across α∈[0,1] | Reachable set is approximately convex near the data manifold |
| Exp 3 sharp degradation at α>1 | Reachable set doesn't extend far beyond the convex hull of real activations |
| Exp 4 sharp transition at f≈0.3 | ~30% of dimensions can be freely set; rest are constrained |
| Exp 4 gradual degradation | No sharp boundary; reachability is more about optimization than geometry |

## Execution Order

Run in this order, since later experiments depend on insights from earlier ones:

1. **Experiment 1** first (calibration; also generates real targets reused in Exp 3 and 4)
2. **Experiment 2a** then **2b** (may want to adjust step count based on Exp 1)
3. **Experiment 3** (uses pairs from Exp 1)
4. **Experiment 4** (uses targets from Exp 1; most informative, save for last
   so you can tune hyperparameters based on earlier runs)

If Experiment 1 shows that 2000 steps isn't enough (loss curves still
declining at step 2000), increase to 5000 everywhere before proceeding.
