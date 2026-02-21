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

## 2026-02-21 18:50 — Experiment 5 Complete (Target Corruption Sweep)

### Results:
| Fraction Replaced | Median Cosine | PCA Dirs Replaced |
|-------------------|---------------|-------------------|
| 0.0 | 0.9960 | 0 (unchanged) |
| 0.1 | 0.9634 | 76 |
| 0.2 | 0.9655 | 153 |
| 0.3 | 0.9611 | 230 |
| 0.5 | 0.9639 | 384 |
| 0.7 | 0.9720 | 537 |
| 1.0 | 0.9647 | 768 |

### Interpretation:
**The corruption curve is NOT monotonically decreasing — it shows a U-shape!**

1. **Sharp initial drop:** From f=0.0 (cos 0.996) to f=0.1 (cos 0.963), there's a large 0.033 drop. Replacing just the top 10% of PCA directions (the 76 highest-variance directions) immediately makes the target much harder.

2. **Plateau from f=0.1 to f=0.5:** Cosine stays around 0.961-0.966 regardless of how many additional directions are corrupted. This is the "floor" for chimeric targets.

3. **Slight recovery at f=0.7:** cos=0.972 — *higher* than f=0.3! When most of the target's structure is replaced, the optimization may be finding it easier because the target is more "generic" (closer to a random target).

4. **f=1.0 (fully random) gives cos=0.965:** comparable to f=0.1. This is consistent with Exp 2b (raw random targets, cos 0.971 with 3000 steps — here we used only 1000 steps, accounting for the small gap).

**Key insight:** The hardest targets to reach are not fully random, but rather partially corrupted — chimeras that have some real structure (constraining the optimization) mixed with inconsistent random components. This is consistent with the theoretical expectation that "contradictory activation regimes" cause difficulty.

**The reachable set has no sharp boundary.** Even at full corruption, cosine remains above 0.96. The "boundary" between reachable and unreachable is gradual, and the transition happens mostly in the first 10% of PCA direction corruption.

## 2026-02-21 19:00 — All Experiments Complete: Synthesis

### Summary of Results

| Experiment | Condition | Median Cosine | Key Finding |
|-----------|-----------|---------------|-------------|
| 1 | Real targets | 0.9960 | Near-perfect reachability |
| 1b | Embedding distance | cos=0.15 | Solutions far from token manifold |
| 2a | Dist-matched random | 0.9771 | Still reachable, but harder |
| 2b | Raw random | 0.9708 | Even random dirs mostly reachable |
| 3 | Interpolation (α=0.5) | 0.9880 | Midpoints slightly harder |
| 3 | Extrapolation (α=2.0) | 0.9731 | Graceful degradation |
| 4a | No FFN | 0.9949 | FFN barely matters (Δ=-0.001) |
| 4b | No Attention | 0.9792 | Attention matters more (Δ=-0.017) |
| 5 | 10% PCA corrupted | 0.9634 | Sharp initial drop |
| 5 | 100% PCA corrupted | 0.9647 | Plateau (no worse than 10%) |

### Answering the Core Research Questions

**Q: Can you find a layer-0 input that produces a target activation at layer 6?**
**A: Yes, almost always, to very high accuracy (cos > 0.96 for all conditions tested).**

The layer-0 to layer-6 map in Pythia-160M is effectively surjective in the directional sense. Given 20 × 768 = 15,360 free parameters targeting 20 × 768 = 15,360 values, the system is well-determined and gradient descent reliably finds solutions.

**Q: How does reachability depend on FFN nonlinearity vs attention mixing?**
**A: Attention mixing is ~17x more important than FFN nonlinearity for reachability.**

- Removing all FFN layers: Δ = -0.001 (negligible)
- Removing all attention layers: Δ = -0.017 (small but consistent)
- The softmax in attention is the only nonlinearity in the no-FFN model, yet it provides sufficient expressivity for cos > 0.994

**Q: How far are optimized inputs from the token embedding manifold?**
**A: Extremely far. This is arguably the most important practical finding.**

- Cosine to nearest token embedding: 0.15 (essentially orthogonal)
- Optimized vector norms: 20.8 (vs embedding norms: 0.8 — a 26x ratio)
- 0% of optimized vectors have cos > 0.9 to any real token

This means:
1. The theoretical "soft prompt attack surface" (anything reachable via unconstrained R^768) is vast
2. But the practical attack surface (anything reachable via token sequences) is unknown and likely much smaller
3. **The unconstrained problem is NOT a good proxy for the constrained problem**

### Implications

**For Safety/Security:**
The good news: while arbitrary mid-layer states are theoretically reachable from unconstrained soft prompts, the solutions lie in regions of embedding space that no real token occupies. A soft prompt API that restricts inputs to the token embedding manifold (or a neighborhood thereof) would likely have a much smaller attack surface. The bad news: we haven't shown this restriction is sufficient — it's a necessary follow-up experiment.

**For Prompt-Based Steering:**
If you want to steer a model to a specific internal state, unconstrained optimization will find a solution, but that solution won't correspond to any real text. Projecting to nearest tokens would likely destroy the carefully optimized state. Constrained optimization (within the token manifold) is the relevant problem for practical steering.

**For Transformer Geometry:**
The first 6 layers of Pythia-160M are extremely expressive as a function from R^(20×768) to R^(20×768). The residual connections preserve dimensionality (no collapse), and the attention mechanism provides sufficient cross-position coupling. The FFN nonlinearities (GELU) add negligible additional expressivity for reachability — the attention softmax alone is sufficient.

**For the Manifold Hypothesis:**
Real activations occupy a thin manifold within R^768. The optimization exploits the full ambient space (high-norm, arbitrary-direction vectors) rather than staying on this manifold. The 26x norm gap between optimized inputs and real embeddings is a quantitative measure of how far outside the data manifold the solutions lie.

### Limitations

1. **Model size:** Pythia-160M is small. Larger models may have different reachability properties (more layers = more computational depth = potentially lower reachability, but also more free parameters per position).

2. **Step count:** Raw random targets (Exp 2b) were still converging at 3000 steps. More optimization budget would likely close the gap between conditions.

3. **Target layer:** We only tested layer 6 (middle). Earlier layers should be easier (fewer transformations); later layers may be harder.

4. **Cosine vs MSE divergence:** While cosine similarity is high across all conditions, MSE varies enormously (0.02 for real targets vs 2.0 for raw random). This means the optimizer finds the right direction but wrong scale. For downstream computation, whether this matters depends on how LayerNorm interacts with the residual stream.

5. **No constrained optimization:** The key follow-up — optimizing within the token embedding manifold — was beyond the scope of this session. This is the experiment that would directly answer the practical safety question.

6. **Single model:** All results are for one model. Generalization to other architectures is unknown.

## 2026-02-21 19:10 — Deeper Statistical Analysis

### Target Difficulty Correlations (Experiment 4)

A key question: do the same targets that are hard for the full model also tend to be hard for the ablated models?

| Comparison | Pearson r | p-value |
|-----------|-----------|---------|
| Full vs No-FFN | 0.022 | 0.879 |
| Full vs No-Attn | 0.030 | 0.837 |
| No-FFN vs No-Attn | **-0.517** | **1.2e-4** |

**Key finding:** Target difficulty is completely uncorrelated between the full model and either ablated variant (r≈0.02). The optimization landscape is so well-behaved that what makes a target "hard" depends entirely on the model structure, not intrinsic properties of the target.

**More striking: No-FFN and No-Attn difficulties are anti-correlated (r=-0.517).** Targets that are easy to reach via attention-only paths are *harder* to reach via FFN-only paths, and vice versa. This suggests attention and FFN mechanisms are **complementary** — they cover different regions of activation space, and a target's location determines which mechanism is better suited to reach it. The full model, having both mechanisms, can always use whichever is more appropriate, explaining why it uniformly outperforms both ablated variants.

### Restart Variance

All conditions have restart variance <1e-5. The optimization landscape appears to have a single dominant basin for each target — no evidence of local minima trapping.

### PCA Analysis (Experiment 1b)

The optimized vectors project poorly onto the token embedding PCA subspace:
- k=384 components (65% embedding variance): reconstruction error = 14.7
- The optimized vectors occupy a fundamentally different subspace of R^768 than token embeddings

### Interpolation Variance (Experiment 3)

Variance increases with distance from the data manifold:
- α=0.0 (real target): std=0.0003
- α=0.50 (midpoint): std=0.0012 (4x larger)
- α=2.0 (far extrapolation): std=0.0034 (11x larger)

This makes geometric sense: near the data manifold, the optimization landscape is well-conditioned (many "nearby" solutions exist). Far from the manifold, solutions are more scattered and target-dependent.
