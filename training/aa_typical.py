"""Per-amino-acid typical-embedding statistics for v4 MAP-GNN.

For each amino acid letter (A, R, N, D, ..., V), compute the mean and
std of the per-residue MutaPLM embedding across all residues of that
type in the training corpus. v4's novelty score is then defined as
``||residue_emb - aa_typical_mean[aa]|| / aa_typical_std[aa]``, which
is high for residues whose embedding deviates from the typical
embedding for their amino acid type — a proxy for "this residue carries
unusual context", which mutations naturally produce.

Computed once per cell (per-folder evaluation), cached at
``<datafolder>/aa_typical.pt``. The cache key is implicit in the
datafolder path; rebuild via ``--force_rebuild``.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable

import torch


# 20 standard amino acids. X stays unhandled (rare; novelty defaults to 0).
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def compute_aa_typical(protein_dict: dict, embedding_key: str = "token_representation",
                       seq_key: str = "seq") -> dict:
    """Walk all proteins in ``protein_dict`` and return per-AA (mean, std) tensors.

    Parameters
    ----------
    protein_dict :
        Output of :func:`training.protein_init_with_embedding.protein_init_with_embedding`,
        dict mapping ``sequence_str -> {seq_key: str, embedding_key: tensor (L, D)}``.
    embedding_key :
        Key into each protein's dict for the per-residue embedding tensor.
        Default ``"token_representation"`` matches v1's existing convention.
    seq_key :
        Key into each protein's dict for the sequence string. Default ``"seq"``.

    Returns
    -------
    dict[str, tuple[Tensor, Tensor]]
        Maps each amino-acid letter present in the corpus to (mean, std)
        tensors of shape (D,). Means/stds computed incrementally so memory
        stays O(20 * D) rather than O(total_residues * D).
    """
    # Sniff the embedding dim from the first protein.
    sample = next(iter(protein_dict.values()))
    embs = sample[embedding_key]
    D = int(embs.shape[-1])

    n: dict = defaultdict(int)
    sum_: dict = defaultdict(lambda: torch.zeros(D, dtype=torch.float64))
    sum_sq: dict = defaultdict(lambda: torch.zeros(D, dtype=torch.float64))

    for prot_data in protein_dict.values():
        seq_str = prot_data[seq_key]
        embs = prot_data[embedding_key].float()
        if embs.shape[0] != len(seq_str):
            raise ValueError(
                f"Embedding length {embs.shape[0]} != sequence length {len(seq_str)} "
                f"for sequence starting {seq_str[:20]!r}. v4 requires per-residue "
                f"embeddings (e.g., MutaPLM); sequence-level embeddings won't work."
            )
        for i, aa in enumerate(seq_str):
            n[aa] += 1
            row = embs[i].to(torch.float64)
            sum_[aa] += row
            sum_sq[aa] += row * row

    aa_typical: dict = {}
    for aa in sum_:
        if n[aa] < 2:
            # Only one observation — std is undefined; skip (residues of this
            # type get novelty=0 at runtime via the per-residue lookup fallback).
            continue
        mean = (sum_[aa] / n[aa]).float()
        var = (sum_sq[aa] / n[aa] - mean.double() ** 2).clamp(min=1e-6)
        std = var.sqrt().float()
        aa_typical[aa] = (mean, std)

    return aa_typical


def per_residue_typical_tensors(seq_str: str, aa_typical: dict, D: int):
    """Build per-residue (mean, std) tensors of shape (L, D) for a single protein.

    Residues whose AA is missing from aa_typical (rare; e.g., 'X' or 'B')
    fall back to mean=0, std=1, which makes their novelty equal to the raw
    embedding norm (large but constant — won't dominate softmax).
    """
    L = len(seq_str)
    mean_t = torch.zeros((L, D), dtype=torch.float32)
    std_t = torch.ones((L, D), dtype=torch.float32)
    for i, aa in enumerate(seq_str):
        if aa in aa_typical:
            m, s = aa_typical[aa]
            mean_t[i] = m
            std_t[i] = s
    return mean_t, std_t


def load_or_compute_aa_typical(cache_path: str, protein_dict: dict,
                               force_rebuild: bool = False,
                               embedding_key: str = "token_representation") -> dict:
    """Cache-aware wrapper. Compute aa_typical and persist to ``cache_path``."""
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Reusing aa_typical cache: {cache_path}")
        return torch.load(cache_path)

    print(f"Computing aa_typical from {len(protein_dict)} proteins...")
    aa_typical = compute_aa_typical(protein_dict, embedding_key=embedding_key)
    print(f"  {len(aa_typical)} amino acid types observed; saving cache to {cache_path}")
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(aa_typical, cache_path)
    return aa_typical
