"""Compute MoLFormer-XL molecular embeddings for the v3 ChemBERT-context path.

Replaces the parquet-loaded `metabolite_features` column with live
inference against IBM's MoLFormer-XL-both-10pct (768-d output, same shape
as ChemBERT, so v3's chembert_proj plumbing works unchanged).

Uses a per-dataset-folder cache (`molformer_emb.pt`) so multiple seeds
reuse the same embedding computation. The cache invalidates whenever the
caller passes ``force_rebuild=True``.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch


def compute_molformer_embeddings(
    smiles_list: Iterable[str],
    model_path: str,
    cache_path: str | None = None,
    force_rebuild: bool = False,
    device: str = "cuda:0",
    batch_size: int = 64,
) -> dict:
    """Return dict mapping SMILES -> (768,) float32 numpy embedding.

    Parameters
    ----------
    smiles_list :
        SMILES strings to embed. Duplicates are deduped before compute.
    model_path :
        Local path to the MoLFormer-XL HuggingFace snapshot
        (e.g., ``.../transformersmodels/ibm/MoLFormer-XL-both-10pct``).
    cache_path :
        Optional disk cache. When set and ``force_rebuild`` is False,
        load from disk if it exists. After computing, write to disk so
        subsequent seeds reuse the result.
    force_rebuild :
        Bypass the cache and recompute. Caller controls invalidation
        (mirrors the train_trace_kin.py --force_rebuild semantics).
    device :
        Device for inference. Falls back to CPU if CUDA unavailable.
    batch_size :
        SMILES per forward pass. 64 fits comfortably in <1 GB of GPU RAM
        for MoLFormer-XL.
    """
    unique_smiles = sorted({s for s in smiles_list if s})

    if cache_path and os.path.exists(cache_path) and not force_rebuild:
        print(f"Reusing MoLFormer cache: {cache_path}")
        cached = torch.load(cache_path)
        missing = [s for s in unique_smiles if s not in cached]
        if not missing:
            return cached
        # Cache exists but is missing some SMILES (e.g. a new pool was
        # added). Compute the deltas and merge.
        print(f"  cache missing {len(missing)} SMILES; computing deltas")
        delta = _embed_with_model(missing, model_path, device, batch_size)
        cached.update(delta)
        torch.save(cached, cache_path)
        return cached

    smi2emb = _embed_with_model(unique_smiles, model_path, device, batch_size)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(smi2emb, cache_path)
        print(f"Saved MoLFormer cache: {cache_path} ({len(smi2emb)} SMILES)")

    return smi2emb


def _embed_with_model(smiles: list[str], model_path: str, device: str,
                      batch_size: int) -> dict:
    """Load MoLFormer-XL once, embed `smiles` in batches, return dict."""
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available() and device.startswith("cuda"):
        print(f"WARN: CUDA requested but unavailable; falling back to CPU")
        device = "cpu"

    print(f"Loading MoLFormer-XL from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, deterministic_eval=True, trust_remote_code=True
    )
    model = model.to(device).eval()

    print(f"Computing MoLFormer embeddings for {len(smiles)} unique SMILES "
          f"(batch_size={batch_size})")
    smi2emb: dict = {}
    for i in range(0, len(smiles), batch_size):
        batch = smiles[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # MoLFormer-XL returns ModelOutput with `pooler_output` of shape
        # (B, hidden_size=768) — sentence-level molecular embedding.
        emb = outputs.pooler_output.detach().cpu().numpy().astype(np.float32)
        for s, e in zip(batch, emb):
            smi2emb[s] = e.reshape(-1)
        if (i // batch_size) % 20 == 0:
            print(f"  embedded {i + len(batch)}/{len(smiles)}")

    # Free the model from GPU before training starts.
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return smi2emb
