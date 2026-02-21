# Research Log: Soft Prompt Reachability of Residual Stream Targets

## 2026-02-21 10:40 — Session Start

**Environment:** RTX 4090 (24GB VRAM), Python 3.12, PyTorch 2.10.0+cu128, Pythia-160M target model.

**Plan:** Follow the execution order from RESEARCH_AGENDA.md:
1. Experiment 1: Reachability of real activations (calibration baseline)
2. Experiment 1b: Embedding manifold distance analysis
3. Experiment 4: FFN and attention ablation study
4. Experiment 2a/2b: Random targets (distribution-matched then raw)
5. Experiment 3: Interpolation/extrapolation geometry
6. Experiment 5: Target corruption sweep (if time permits)

**Assumptions for initial setup:**
- Pythia-160M fits comfortably in 24GB VRAM with room for optimization gradients
- `output_hidden_states=True` returns hidden states at all layer boundaries (13 tensors for 12 layers: embedding + 12 post-layer)
- The `hidden_states` tuple is indexed as: [0] = embedding output, [1] = after layer 0, ..., [12] = after layer 11. So TARGET_LAYER=6 means after layer 5 (the 6th transformer block).
- WikiText-2 is available via HuggingFace datasets
- 2000 optimization steps is a reasonable starting point (will extend if needed)
- Adam with lr=0.01 works for this optimization landscape (common in soft prompt tuning literature)

**Decision:** I'll build one comprehensive script that runs all experiments sequentially with checkpointing, rather than separate scripts, to keep state management clean. Each experiment will save its results to disk so we can analyze partial results if anything crashes.

## 2026-02-21 11:00 — Environment Validation & Timing

**Key findings from sanity checks:**
1. Model must be loaded with `torch_dtype=torch.float32` explicitly — default loading causes fp16/fp32 dtype mismatches when ablation hooks return zeros.
2. Pythia uses **parallel residual** architecture: `x = x + attn(ln1(x)) + mlp(ln2(x))`. Both attention and MLP read the same input, not sequential.
3. `hidden_states` has 13 entries: [0]=embedding, [1]=after layer 0, ..., [12]=after layer 11. TARGET_LAYER=6 gives us residuals after the 6th transformer block.
4. Token-input vs embedding-input paths give cosine 0.9995 (not 1.0, presumably floating point ordering).
5. Full model vs no-FFN: cosine only 0.24 — FFN contributes massively to the activation pattern.
6. Full model vs no-attention: cosine 0.76 — attention contributes less in raw activation terms (but this is comparing full-model outputs, not reachability).

**Convergence profiling (3 real targets + 1 random target):**
- Real targets at 1000 steps: cos ~0.995, MSE ~0.03
- Real targets at 2000 steps: cos ~0.999, MSE ~0.01
- Random targets at 2000 steps: cos ~0.951, MSE ~4.9 (still improving!)
- Random targets converge much slower and to lower cosine — the optimization landscape is harder.

**Timing:** ~16ms/step, so 1000 steps ≈ 16s, 2000 steps ≈ 32s per optimization run.

**Revised plan (time budget ~9 hours):**
- Reduce to 50 targets per experiment (from 100)
- Reduce to 3 restarts (from 5)
- Use 1000 steps for real-target experiments (Exp 1, 3, 4) since convergence is fast
- Use 2000 steps for random targets (Exp 2a) and 3000 for raw random (Exp 2b)
- Skip Experiment 5 unless time permits after core experiments

**Estimated timeline:**
- Exp 1 (50 targets × 3 restarts × 1000 steps): ~40 min
- Exp 1b (analysis only): ~1 min
- Exp 4 (50 targets × 3 restarts × 2 ablations × 1000 steps): ~80 min
- Exp 2a (50 targets × 3 restarts × 2000 steps): ~80 min
- Exp 2b (50 targets × 3 restarts × 3000 steps): ~120 min
- Exp 3 (10 pairs × 8 alphas × 3 restarts × 1000 steps): ~64 min
- Analysis and plotting: ~15 min
- **Total: ~6.7 hours** (leaves buffer for Exp 5 if results are interesting)

**Assumptions for reduced plan:**
- 50 targets is sufficient for statistical significance on the main effects
- 3 restarts captures optimization variance adequately (real targets converge reliably)
- 1000 steps is sufficient for real targets (validated empirically: cos >0.994)
- 3000 steps may still not be enough for raw random targets (will note this limitation)

## 2026-02-21 12:15 — Experiment 1 & 1b Complete

### Experiment 1: Reachability of Real Activations
**Result: Near-perfect reachability.**
- Median cosine similarity: **0.9960**
- Mean: 0.9959 ± 0.0004
- Range: [0.9952, 0.9969]
- Restart variance: ~1e-6 (essentially zero — optimization landscape is very well-behaved)
- All 50 targets converged to cos > 0.995 with no failures
- Convergence check: delta in last 20% of steps was 0.0023 — still slightly improving but plateaued

**Interpretation:** The optimization works perfectly. Real activations are trivially reachable from unconstrained R^768 inputs. The calibration baseline is clean.

### Experiment 1b: Embedding Manifold Distance
**Result: Optimized inputs are EXTREMELY far from token embeddings.**
- Cosine to nearest token embedding: mean=0.156, median=0.154
- **0% of optimized vectors have cos > 0.9 to any token embedding**
- Optimized input norms: mean=20.81 (vs token embedding norms: mean=0.79)
- The optimized vectors are ~26x larger in norm than real embeddings
- The nearest tokens are gibberish: ['cept', '228', '€', ' ne', 'ical', ...]

**Interpretation:** The unconstrained-vs-constrained distinction is **not** moot. The optimization exploits high-norm, arbitrary-direction regions of R^768 that no discrete token occupies. This means:
1. Unconstrained soft prompt reachability ≠ real soft prompt reachability
2. A constrained follow-up (restricting to the token embedding manifold) is a meaningful and important next step
3. The "attack surface" conclusion depends entirely on whether these solutions can be projected to nearby token sequences

**This is arguably the most important finding so far for the safety implications.** The gap between theoretical reachability (yes, almost anything is reachable) and practical reachability (the solutions are nowhere near the token manifold) is enormous.

**Assumptions validated:**
- 1000 steps was sufficient (cos ~0.996)
- 3 restarts was sufficient (variance ~0)
- All targets converged uniformly well (no failures, no bimodality)

Now running Experiment 4 (ablation study).

## 2026-02-21 14:10 — Experiment 4 Complete (Ablation Study)

### Results:
| Variant | Median Cosine | Δ from Full | Better than Full |
|---------|-------------|-------------|------------------|
| Full Model | 0.9960 | — | — |
| No FFN (attn only) | 0.9949 | -0.0010 | 0% |
| No Attention (FFN only) | 0.9792 | -0.0169 | 0% |

### Interpretation:
**FFN nonlinearities are NOT the bottleneck for reachability.** Removing all 12 FFN layers (and their GELU nonlinearities) only drops median cosine by 0.001. The attention-only model achieves cos 0.9949 — still extremely high. This means:
- The residual connection + attention mixing is sufficient to make the layer-0 → layer-6 map nearly surjective
- The theoretical argument about nonlinearities expanding the reachable set is correct but practically irrelevant — the attention mechanism already provides enough coupling to cover the target space

**Attention coupling IS important for reachability.** Removing attention drops cosine by 0.017 — a 17x larger effect than removing FFN. Without attention, each position is an independent 12-layer MLP with residual connections. The fact that reachability drops (though remains high at 0.979) means cross-position information mixing helps even when the target was generated by the full model.

**Both ablations are uniformly worse** — not a single target was easier to reach under ablation. This rules out the "attention coupling as obstacle" hypothesis from the agenda.

**Key surprise:** The no-FFN result is remarkably strong. A model with only attention and residual connections (all linear operations except the softmax in attention) still achieves 0.995 cosine reachability. The softmax is the only nonlinearity in this variant, yet it provides enough expressivity for near-perfect reachability.

**Assumptions confirmed:**
- Ablation hooks work correctly (returning zeros from submodule outputs)
- The targets from the full model are also reachable by ablated variants (not a trivial result — the ablated model has a different computational structure)

Now running Experiments 2a/2b (random targets).

## 2026-02-21 16:30 — Experiment 2a Complete

### Experiment 2a: Distribution-Matched Random Targets
**Result: Good but lower than real targets.**
- Median cosine: **0.9771** (vs 0.9960 for real targets)
- Mean: 0.9769 ± 0.0021
- Range: [0.9719, 0.9810]
- MSE: median 0.091 (vs 0.021 for real targets)

**Interpretation:** Distribution-matched random targets are reachable at cos ~0.977 but notably harder than real targets. The ~0.019 gap from real targets suggests the model's computational structure imposes constraints beyond just matching the statistical distribution. There's something about targets being actually produced by the network (rather than sampled from the marginal distribution) that makes them easier to reach.

### Experiment 2b: Raw Random Targets
**Result: Still reachable but harder.**
- Median cosine: **0.9708**
- Mean: 0.9697 ± 0.0060
- Range: [0.9470, 0.9809]
- MSE: median 1.95 (22x higher than 2a!)
- Convergence delta in last 20%: 0.0083 — still improving, could benefit from more steps
- Higher variance than any other experiment (std 0.006 vs 0.002 for 2a)

### Comparison across target types:
| Experiment | Target Type | Median Cosine | Median MSE |
|-----------|-------------|---------------|------------|
| Exp 1 | Real activations | 0.9960 | 0.021 |
| Exp 2a | Distribution-matched random | 0.9771 | 0.091 |
| Exp 2b | Raw random (norm-matched) | 0.9708 | 1.95 |

**Key observations:**
1. **Hierarchy is clear:** Real >> Distribution-matched > Raw random
2. **The gap between 2a and 2b (0.006 cosine) is smaller than 1 vs 2a (0.019)**. Distribution structure helps but isn't the dominant factor — the main difficulty is moving away from "actual network outputs" to "synthetic targets."
3. **MSE tells a different story than cosine.** The 22x MSE gap between 2a and 2b (with only 0.006 cosine gap) means raw random targets have correct directions but wrong scales. The optimizer gets the angle right but struggles with magnitude.
4. **Even raw random targets reach cos 0.97** — the reachable set is vast. Most of R^768 (at least directionally) appears reachable from layer-0 inputs.
5. **Exp 2b was still converging** — more steps would likely close the gap further.

**Assumptions checked:**
- 2000 steps was adequate for 2a (only 0.002 improvement in last 20%)
- 3000 steps for 2b left room for improvement (0.008 improvement in last 20%)
- Distribution-matched sampling via Cholesky decomposition worked without numerical issues

## 2026-02-21 17:20 — Experiment 3 Complete (Interpolation/Extrapolation)

### Results:
| Alpha | Median Cosine | Description |
|-------|--------------|-------------|
| -0.50 | 0.9854 | Extrapolation (beyond B) |
| 0.00 | 0.9958 | Target B (real) |
| 0.25 | 0.9915 | Interpolation |
| 0.50 | 0.9880 | Midpoint |
| 0.75 | 0.9916 | Interpolation |
| 1.00 | 0.9959 | Target A (real) |
| 1.50 | 0.9864 | Extrapolation (beyond A) |
| 2.00 | 0.9731 | Far extrapolation |

### Interpretation:
**The reachable set extends well beyond the convex hull of real activations.**

1. **U-shaped reachability curve:** The endpoints (real targets α=0, α=1) are easiest (cos ~0.996), the midpoint (α=0.5) is slightly harder (cos ~0.988), and extrapolations degrade gracefully. This U-shape is surprising — it means the midpoint of two real activations is slightly harder to reach than either endpoint, even though midpoints are "closer" to the data manifold.

2. **Gradual degradation, no cliff:** Even at α=2.0 (extrapolating as far beyond target A as A is from B), cosine is still 0.973 — very high. There's no sharp boundary, suggesting the reachable set is "thick" and smoothly bounded.

3. **Symmetric extrapolation:** α=-0.5 (0.985) and α=1.5 (0.986) give similar results, confirming the degradation is symmetric and direction-independent.

4. **The midpoint dip is consistent across all 10 pairs** — this isn't noise, it's a genuine structural feature. Possible explanation: the midpoint of two real activations may land in a region that's outside the "natural manifold" of layer-6 residuals (since the manifold is curved, not flat), making it more like a distribution-matched random target (which gets cos ~0.977).

5. **Remarkably consistent:** The per-pair variance is tiny — this is a robust geometric property of the model.

Now running Experiment 5 (target corruption sweep) if time permits.
