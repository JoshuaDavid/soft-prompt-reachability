#!/usr/bin/env python3
"""
Soft Prompt Reachability Demo
=============================

Demonstrates four key findings about steering transformer internals
via optimized inputs, using Pythia-160M as a test model.

Finding 1: Unconstrained soft prompts reach arbitrary mid-layer activations (cos≈0.996)
Finding 2: Projecting to nearest tokens destroys this (cos≈0.60)
Finding 3: Token-mixture optimization recovers it with real tokens (cos≈0.96)
Finding 4: A short prefix can perfectly steer distant positions (cos≈1.0)

Runtime: ~5 minutes on a GPU.
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Setup ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "EleutherAI/pythia-160m"
TARGET_LAYER = 6  # Middle of 12-layer model
SEQ_LEN = 20

print(f"Loading {MODEL} on {DEVICE}...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model.eval()
for p in model.parameters():
    p.requires_grad = False
model.to(DEVICE)
embed_matrix = model.gpt_neox.embed_in.weight  # [vocab_size, 768]
D = model.config.hidden_size  # 768
print(f"Ready: {model.config.num_hidden_layers} layers, d_model={D}\n")


def get_residual(input_embeds, layer=TARGET_LAYER):
    """Forward pass through the first `layer` transformer blocks."""
    h = input_embeds.unsqueeze(0)
    pos_ids = torch.arange(h.shape[1], device=DEVICE).unsqueeze(0)
    pos_emb = model.gpt_neox.rotary_emb(h, position_ids=pos_ids)
    for i in range(layer):
        h = model.gpt_neox.layers[i](h, position_embeddings=pos_emb)
    return h.squeeze(0)


def cos_sim(a, b):
    """Mean per-position cosine similarity."""
    return F.cosine_similarity(a, b, dim=-1).mean()


def load_text_tokens(n, seq_len):
    """Load n token sequences from WikiText-2."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    seqs = []
    for ex in ds:
        text = ex["text"].strip()
        if len(text) < 20:
            continue
        toks = tokenizer.encode(text, return_tensors="pt").squeeze(0)
        if toks.shape[0] >= seq_len:
            seqs.append(toks[:seq_len].to(DEVICE))
        if len(seqs) >= n:
            break
    return seqs


# ── Get a real target activation to aim for ──────────────────────────

print("Loading real text targets...")
seqs = load_text_tokens(5, SEQ_LEN)
tokens_a, tokens_b = seqs[0], seqs[1]

with torch.no_grad():
    target = get_residual(model.gpt_neox.embed_in(tokens_a.unsqueeze(0)).squeeze(0))


# ═══════════════════════════════════════════════════════════════════════
# Finding 1: Unconstrained optimization reaches the target
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("FINDING 1: Unconstrained soft prompt optimization")
print("=" * 60)

torch.manual_seed(0)
prompt = torch.randn(SEQ_LEN, D, device=DEVICE, requires_grad=True)
opt = torch.optim.Adam([prompt], lr=0.01)

for step in range(1000):
    opt.zero_grad()
    residual = get_residual(prompt)
    loss = 1 - cos_sim(residual, target) + 0.01 * ((residual - target) ** 2).mean()
    loss.backward()
    opt.step()

with torch.no_grad():
    final_cos = cos_sim(get_residual(prompt), target).item()

print(f"  After 1000 steps: cosine similarity = {final_cos:.4f}")
print(f"  → Unconstrained soft prompts reach arbitrary activations.\n")


# ═══════════════════════════════════════════════════════════════════════
# Finding 2: Projecting to nearest tokens destroys the match
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("FINDING 2: Nearest-token projection")
print("=" * 60)

with torch.no_grad():
    optimized = prompt.detach()

    # Project each position to its nearest token embedding
    sims = F.normalize(optimized, dim=-1) @ F.normalize(embed_matrix, dim=-1).T
    nearest_ids = sims.argmax(dim=-1)
    projected = embed_matrix[nearest_ids]

    proj_cos = cos_sim(get_residual(projected), target).item()

    # Random token baseline
    random_ids = torch.randint(0, embed_matrix.shape[0], (SEQ_LEN,), device=DEVICE)
    random_cos = cos_sim(get_residual(embed_matrix[random_ids]), target).item()

print(f"  Optimized continuous:     cosine = {final_cos:.4f}")
print(f"  Nearest-token projection: cosine = {proj_cos:.4f}")
print(f"  Random tokens:            cosine = {random_cos:.4f}")
print(f"  → Projecting to nearest tokens loses almost everything.\n")


# ═══════════════════════════════════════════════════════════════════════
# Finding 3: Token-mixture optimization recovers discrete reachability
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("FINDING 3: Token-mixture optimization (discrete tokens)")
print("=" * 60)

vocab_size = embed_matrix.shape[0]
tau = 0.5  # Temperature: balances exploration vs discreteness

torch.manual_seed(0)
logits = torch.randn(SEQ_LEN, vocab_size, device=DEVICE) * 0.01
logits.requires_grad_(True)
opt = torch.optim.Adam([logits], lr=0.05)

for step in range(2000):
    opt.zero_grad()
    weights = F.softmax(logits / tau, dim=-1)       # Soft token selection
    soft_embeds = weights @ embed_matrix              # Weighted mixture of embeddings
    residual = get_residual(soft_embeds)
    loss = 1 - cos_sim(residual, target) + 0.01 * ((residual - target) ** 2).mean()
    loss.backward()
    opt.step()

with torch.no_grad():
    # Continuous result (soft mixture)
    weights = F.softmax(logits / tau, dim=-1)
    cont_cos = cos_sim(get_residual(weights @ embed_matrix), target).item()

    # Discrete result (argmax → actual tokens)
    token_ids = logits.argmax(dim=-1)
    discrete_cos = cos_sim(get_residual(embed_matrix[token_ids]), target).item()
    chosen_text = tokenizer.decode(token_ids.tolist())

print(f"  Continuous (soft mixture): cosine = {cont_cos:.4f}")
print(f"  Discrete (argmax tokens):  cosine = {discrete_cos:.4f}")
print(f"  Tokens found: {repr(chosen_text[:80])}...")
print(f"  → Optimizing within the token manifold finds real tokens that work.\n")


# ═══════════════════════════════════════════════════════════════════════
# Finding 4: A short prefix can steer distant positions perfectly
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("FINDING 4: Prefix steering (20-token prefix, 100-token context)")
print("=" * 60)

PREFIX_LEN = 20
CONTEXT_LEN = 100


def get_residual_long(embeds, layer=TARGET_LAYER):
    """Forward for longer sequences."""
    h = embeds.unsqueeze(0)
    pos_ids = torch.arange(h.shape[1], device=DEVICE).unsqueeze(0)
    pos_emb = model.gpt_neox.rotary_emb(h, position_ids=pos_ids)
    for i in range(layer):
        h = model.gpt_neox.layers[i](h, position_embeddings=pos_emb)
    return h.squeeze(0)


# Load a longer context
long_seqs = load_text_tokens(5, CONTEXT_LEN)
context_tokens = long_seqs[0]
context_embeds = model.gpt_neox.embed_in(context_tokens.unsqueeze(0)).squeeze(0).detach()

# Target: a different context's activation at the last position
other_context = long_seqs[1]
other_embeds = model.gpt_neox.embed_in(other_context.unsqueeze(0)).squeeze(0).detach()
with torch.no_grad():
    pad = torch.zeros(PREFIX_LEN, D, device=DEVICE)
    target_act = get_residual_long(torch.cat([pad, other_embeds]))[PREFIX_LEN + CONTEXT_LEN - 1]

# Optimize the prefix to steer position 119 (last position, 99 tokens from prefix)
torch.manual_seed(0)
prefix = torch.randn(PREFIX_LEN, D, device=DEVICE) * 0.1
prefix.requires_grad_(True)
opt = torch.optim.Adam([prefix], lr=0.01)

for step in range(2000):
    opt.zero_grad()
    full = torch.cat([prefix, context_embeds])
    achieved = get_residual_long(full)[PREFIX_LEN + CONTEXT_LEN - 1]
    loss = 1 - F.cosine_similarity(achieved.unsqueeze(0), target_act.unsqueeze(0))
    loss.backward()
    opt.step()

with torch.no_grad():
    full = torch.cat([prefix.detach(), context_embeds])
    achieved = get_residual_long(full)[PREFIX_LEN + CONTEXT_LEN - 1]
    steer_cos = F.cosine_similarity(achieved.unsqueeze(0), target_act.unsqueeze(0)).item()

print(f"  Steering position 119 (distance 99 from prefix): cosine = {steer_cos:.4f}")
print(f"  → A 20-token prefix perfectly steers activations 99 positions away.\n")


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  1. Unconstrained optimization:  cos = {final_cos:.4f}   (near-perfect)")
print(f"  2. Nearest-token projection:    cos = {proj_cos:.4f}   (destroyed)")
print(f"  3. Token-mixture optimization:  cos = {discrete_cos:.4f}   (recovered)")
print(f"  4. Prefix steering at dist 99:  cos = {steer_cos:.4f}   (perfect)")
print()
print("  Key insight: the residual stream is almost fully reachable via")
print("  soft prompt optimization — and this extends to real discrete")
print("  tokens when optimized correctly (but not via naive projection).")
print("  A short prefix can steer activations at arbitrary distances.")
