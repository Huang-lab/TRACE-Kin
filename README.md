# TRACE-Kin — Kinetic Parameter Prediction Backbone

Kinetic-parameter prediction backbone of **TRACE** (Transformative Reasoning Across Catalytic Enzymes), a graph-to-reasoning framework for enzymatic drug discovery targeting *Nature Computational Science*. TRACE-Kin is Section 1 of the paper and the foundation that the rest of the pipeline (TRACE-Gen, TRACE-Reason, application case studies) consumes from.

> **The single source of truth for this repo is [PROJECT.md](PROJECT.md).** Read it for full status, architecture details, the v2 redesign, the promotion gate, and HPC reproduction recipes.

## What this repo produces

Three artifacts feed downstream TRACE components:

1. **Kinetic predictions** — Km, kcat, Ki, Kd, and kcat/Km — feed TRACE-Reason's interpretation block and gate biocatalysis case feasibility.
2. **Cross-attention scores** between drug atoms and protein residues — the raw signal TRACE-Gen's reranker uses to identify sites of metabolism, and the physical evidence TRACE-Reason cites in its reasoning traces.
3. **Interaction fingerprints** (400/600-dim concatenation) — the per-pair embedding application cases use to rank enzyme/substrate combinations.

## Status

* **v1** — PSICHIC-adapted architecture (PNA + MinCut pooling + cross-attention) is **frozen**. Its 1,847-row benchmark in [trace_doc/kinetic_regress_benchmark.csv](trace_doc/kinetic_regress_benchmark.csv) is read-only ground truth.
* **v2 redesign** — implemented in [models/trace_kin/net_v2.py](models/trace_kin/net_v2.py); targets the structural reasons Random Forest beats v1 on catalytic kinetics (information bottleneck, self-derived contact-map noise, MinCut-loss misalignment). The v2 rerun on 14 selected datasets is ready to submit; results decide whether paper figures are produced.
* See [PROJECT.md §2 / §6](PROJECT.md) for full status and the RF gap analysis.

## Repo layout (top-level)

| Path | Role |
|---|---|
| [`PROJECT.md`](PROJECT.md) | Source of truth for this repo |
| [`trace_doc/`](trace_doc/) | **Immutable** paper blueprint — never edit |
| [`models/trace_kin/`](models/trace_kin/) | v1 (`net_v1.py`) and v2 (`net_v2.py`) architectures |
| [`training/`](training/) | Training entry point, trainer (with SWA), datasets, LSF arrays |
| [`analysis/`](analysis/) | Cleanup, tables, significance tests, v2 aggregation, promotion-gate scripts |
| [`figures/`](figures/) | Figure generation |
| [`inference/`](inference/) | Predictor and dual-embedding confidence (P2 deliverables) |
| [`data/benchmark/`](data/) | Cleaned benchmark CSV, normalization log, tables, gate decisions |

## Quick start

For laptop development of analysis/figures/inference scripts only (no GPU/training):

```bash
pip install pandas numpy scipy matplotlib scikit-learn lifelines
```

For full training (GPU/CPU/macOS conda environments are inherited from upstream PSICHIC):

```bash
# Linux/Windows GPU
conda env create -f environment_gpu.yml
# macOS
conda env create -f environment_osx.yml
# Linux/Windows CPU
conda env create -f environment_cpu.yml
```

See [PROJECT.md §10](PROJECT.md) for full local-development setup and [PROJECT.md §9](PROJECT.md) for HPC submission patterns (LSF arrays via `bsub < file.lsf`, chained with `-w "done(<ID>)"`).

## Smoke test the model code

```bash
python -m py_compile models/trace_kin/net_v1.py models/trace_kin/net_v2.py training/trainer.py training/train_trace_kin.py
```

## Attribution

TRACE-Kin is adapted from **PSICHIC** (Koh et al., *Nature Machine Intelligence*, 2024 — [paper](https://www.nature.com/articles/s42256-024-00847-1)). The v1 architecture in `models/trace_kin/net_v1.py` is a frozen baseline derived from PSICHIC; the v2 redesign in `net_v2.py` introduces the embedding shortcut, attention pooling, embedding-dominant residue fusion, SWA, and cross-dataset training pooling. The original PSICHIC README, tutorials, and demo assets are preserved locally in `to_remove/legacy_psichic_assets/` for reference and credit.

```
PSICHIC: physicochemical graph neural network for learning protein-ligand
interaction fingerprints from sequence data
Huan Yee Koh, Anh T.N. Nguyen, Shirui Pan, Lauren T. May, Geoffrey I. Webb
Nature Machine Intelligence (2024)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
