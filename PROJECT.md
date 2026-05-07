# TRACE-Kin — Kinetic Parameter Prediction Backbone

> **Last updated:** 2026-05-07. Single source of truth for this repo. The
> blueprint that this repo implements lives in
> [trace_doc/](trace_doc/) and is **immutable** — do not edit anything inside
> it. This document supersedes the now-archived `HPC_TRAINING_GUIDE.md`,
> `TRAINING_GUIDE.md`, `PSICHIC_Training_Analysis.md`,
> `PSICHIC_Kin_Next_Phase_Decision.md`, and `Tier1_Rerun_Jobs.md`.

---

## 1. The kinetic-prediction backbone of TRACE

This repo is the **kinetic parameter prediction backbone** for the entire
TRACE pipeline (Transformative Reasoning Across Catalytic Enzymes), targeting
Nature Computational Science. It is Section 1 in the paper structure, but
operationally it sits at the foundation of the rest:

```
TRACE-Kin (this repo)  ─┬─►  TRACE-Gen (reaction prediction + graph reranker)
                        ├─►  Applications (CYP pharmacogenomics, biocatalysis,
                        │    cancer enzyme targeting)
                        └─►  TRACE-Reason (multimodal LLM)
```

Three concrete artifacts produced here are consumed downstream:

* **Kinetic-parameter predictions** (kcat, Km, Ki, Kd, kcat/Km) — feed
  TRACE-Reason's kinetic-interpretation block and gate biocatalysis case
  feasibility.
* **Cross-attention scores** between drug atoms and protein residues — the
  raw signal TRACE-Gen's reranker uses to identify sites of metabolism, and
  the physical evidence TRACE-Reason cites in its reasoning traces.
* **Interaction fingerprints** (the 400/600-dim concatenation feeding the
  regression head) — the per-pair embedding the application cases use to
  rank enzyme/substrate combinations.

Without this backbone working, the rest of TRACE has no graph-grounded
substrate to reason from. The "graph-to-reasoning" thesis in
[trace_doc/TRACE_research_design.md](trace_doc/TRACE_research_design.md)
requires this backbone to produce competitive predictions — currently v1
does not on most catalytic kinetics (see §6), which is the core problem
the v2 redesign tackles. See
[trace_doc/TRACE_implementation_plan.md](trace_doc/TRACE_implementation_plan.md)
section 1A for the official next-step list this repo executes.

---

## 2. Status (2026-05-07)

* **Historical benchmark frozen.** `trace_doc/kinetic_regress_benchmark.csv`
  contains 1,847 (model × embedding × kinetic × split × dataset) test
  results. This is the v1 baseline; treat it as read-only ground truth for
  what PSICHIC-style models achieve on these tasks.
* **Tier 1 / Tier 2 hyperparameter ablation halted.** The Phase 1
  validation reruns regressed (Phase 1a: 0.876 → 1.1875 RMSE; Phase 2: 5
  of 6 regressed or below the 0.02 promotion gate). The next-phase gate
  decision shut Phase 3 down. All ablation scaffolding is archived under
  `to_remove/tier1_failed/`.
* **v2 redesign in progress.** This rebuild adds a new architecture
  (`models/trace_kin/net_v2.py`) targeting the structural reasons RF beats
  PSICHIC on catalytic tasks. The v2 rerun on 14 selected datasets is
  ready to submit (`training/rerun_12_datasets.sh`); results decide whether
  paper figures are produced.
* **Packaging deliverables prepared.** CSV cleanup, table generation,
  figure generation, significance testing, and v2-vs-v1 aggregation are
  all wired up but **gated** on the promotion-gate verdict.

---

## 3. Repo layout

```
TRACE_Kin/
├── trace_doc/                          # IMMUTABLE — paper blueprint
│   ├── TRACE_research_design.md
│   ├── TRACE_implementation_plan.md
│   └── kinetic_regress_benchmark.csv   # 1,847-row source benchmark
│
├── PROJECT.md                          # this file
├── README.md                           # upstream PSICHIC attribution (kept for license/credit)
├── LICENSE
├── environment_{gpu,cpu,osx}.yml       # original PSICHIC envs
│
├── data/
│   └── benchmark/
│       ├── kinetic_regress_benchmark_clean.csv
│       ├── normalization_log.md
│       ├── tables/                     # main + supplementary tables (md + tex)
│       ├── significance.csv            # Wilcoxon test results
│       ├── trace_kin_v2_results.csv    # v2 rerun aggregated (after HPC run)
│       └── promotion_gate_decision.md  # PASS/FAIL verdict
│
├── models/
│   └── trace_kin/                      # both architecture variants
│       ├── __init__.py
│       ├── net_v1.py                   # frozen baseline (renamed from PSICHIC's `net`)
│       ├── net_v2.py                   # redesigned architecture (Phase 3 of this plan)
│       ├── layers.py
│       ├── protein_pool.py             # MinCut pooling — used by v1 only
│       ├── drug_pool.py
│       ├── pna.py
│       └── scaler.py
│
├── analysis/                           # P0/P1 packaging scripts
│   ├── clean_benchmark.py
│   ├── generate_tables.py
│   ├── significance_tests.py
│   ├── aggregate_v2_results.py
│   └── promotion_gate.py
│
├── figures/
│   ├── generate_figures.py
│   └── output/                         # generated figures (pdf + png)
│
├── inference/                          # P2 deliverables
│   ├── trace_kin_predictor.py
│   └── confidence.py
│
├── training/                           # slim, focused training pipeline
│   ├── train_trace_kin.py              # main entry; supports --model_version v1|v2
│   ├── trainer.py                      # adds SWA support to the legacy Trainer
│   ├── dataset.py                      # ProteinMoleculeDataset
│   ├── data_utils.py                   # DataLoader, samplers, virtual_screening
│   ├── ligand_init.py                  # SMILES → molecular graph + clique tree
│   ├── protein_init_with_embedding.py  # AA features + contact-map graph
│   ├── protein_init.py                 # ESM2-at-runtime variant (kept for completeness)
│   ├── metrics.py
│   ├── config_v1.json                  # exact config that produced the historical benchmark
│   ├── config_v2.json                  # v2 redesign config
│   ├── run_benchmark.lsf               # parametric LSF launcher
│   └── rerun_12_datasets.sh            # submits the 14 v2 rerun jobs
│
└── to_remove/                          # archive (do not delete; used for reference)
    ├── legacy_psichic_assets/          # upstream PSICHIC tutorials, weights, demos
    ├── tier1_failed/                   # halted Tier 1 ablation scaffolding
    ├── legacy_entry_points/            # main.py, train_psichic.py, screening.py
    ├── legacy_training_pre_rebuild/    # the train_psichic_embedding.py era
    └── legacy_docs/                    # the 5 source MDs merged into this file
```

---

## 4. TRACE-Kin v1 architecture (frozen baseline)

> Authoritative summary of what produced the 1,847-row benchmark. Source:
> the original `PSICHIC_Training_Analysis.md` §1.5 and §1.8 (now archived).

### Input encoding

| Component | Input dim | Output dim | Module |
|---|---|---|---|
| Atom type | 20 classes | 200 | `Embedding(20, 200)` |
| Atom features | 43 | 200 | `MLP([43, 400, 200])` |
| Clique type | 4 classes | 200 | `Embedding(4, 200)` |
| Protein AA features | 33 | 200 | `MLP([33, 400, 200])` |
| Protein embedding | varies | 200 | `MLP([emb_dim, 400, 200])` |

Protein fusion is **additive**: `residue_x = prot_aa(residue_aa) + prot_evo(residue_evo)`.
This is one of the structural choices the v2 redesign changes.

### Interaction layers (×3)

Each layer runs:
1. Drug PNAConv (PNA, 5 heads, aggregators=mean/min/max/std,
   scalers=identity/amp/linear).
2. Protein PNAConv (same architecture as drug).
3. MotifPool (atom → clique).
4. GCNCluster + MinCut pooling (residues → K clusters, K=[5, 10, 20]).
   This produces auxiliary `ortho_loss` and `cluster_loss`.
5. DrugProteinConv cross-attention between drug cliques and protein
   clusters.
6. Residual updates: clique→atom, cluster→residue.
7. GraphNorm.

### Output head

`reg_out = MLP([400, 200, 1])` over the concatenated mol + protein pooled
features (200 + 200 = 400).

### Loss

`total = mse(reg_pred, reg_y) + 1.0 * ortho_loss + 1.0 * cluster_loss`. The
auxiliary losses sit at fixed weight 1.0; the Tier 1 ablation tried to tune
LR / dropout / patience instead of these and failed.

### Hyperparameters (matches `training/config_v1.json` exactly)

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| LR scheduling | disabled |
| Weight decay | 1e-4 |
| Gradient clip | 1.0 |
| Dropout (GNN path) | 0 |
| Dropout (cross-attention) | 0.2 |
| Epochs | 100 |
| Batch size | 16 |
| Hidden channels | 200 |
| Interaction layers | 3 |
| Cluster sizes K | [5, 10, 20] |
| Attention heads | 5 |
| Optimizer | AdamW |
| Auxiliary loss weights | 1.0 (fixed) |

---

## 5. TRACE-Kin v2 architecture (redesign)

The v2 architecture targets three structural mechanisms by which Random
Forest beats PSICHIC on catalytic kinetics. See §6 for the full root-cause
analysis. The implementation is in
[`models/trace_kin/net_v2.py`](models/trace_kin/net_v2.py).

### Change 1: Embedding shortcut branch

A parallel path that skips the GNN entirely:

```python
# At the END of forward, beside the GNN-pooled prot_pool_feat:
embedding_pool = scatter(residue_evo_x, prot_batch, dim=0, reduce='mean')   # (B, 1280)
embedding_shortcut = self.shortcut_proj(embedding_pool)                     # (B, 200)
mol_prot_feat = torch.cat([mol_pool_feat, prot_pool_feat, embedding_shortcut], dim=-1)
```

`self.shortcut_proj = MLP([1280, 400, 200])`. Regression head input grows
from 400 → 600. **This is the load-bearing change.** It guarantees the
model has at least the information RF uses on the same embeddings,
eliminating the bottleneck imposed by v1's `prot_evo: 1280 → 200` projection
before fusion.

### Change 2: Attention-pooled cluster aggregation (drops MinCut)

Replaces `dense_mincut_pool` and its `o_loss` / `cl_loss` outputs with a
plain softmax-weighted aggregation:

```python
s_logits = self.cluster[idx](residue_x, dropped_residue_edge_index)  # (N, K)
s_dense, _ = to_dense_batch(s_logits, prot_batch)                     # (B, max_N, K)
s_soft = F.softmax(s_dense, dim=-1) * residue_mask.unsqueeze(-1).float()
cluster_x = torch.bmm(s_soft.transpose(1, 2), residue_hx)              # (B, K, hidden)
```

The K-cluster structure is preserved so `DrugProteinConv` is unchanged.
Auxiliary losses are returned as zero tensors for trainer compatibility.

### Change 3: Embedding-dominant residue fusion

```python
# v1: residue_x = prot_aa(x) + prot_evo(x)
# v2:
residue_x = self.prot_evo(residue_evo_x) + 0.1 * self.prot_aa(residue_aa_x)
```

The `0.1` weight (`aa_residual_weight` in config) is fixed, not learned.
The embedding now dominates (matching what RF uses) while AA features
remain available as a small physicochemical correction.

### Stochastic Weight Averaging

Enabled by default in `training/config_v2.json`. SWA accumulates a running
average of model weights across the final 25% of epochs and substitutes
that average for the best per-epoch checkpoint **only if it evaluates
better on the validation set**. Implementation lives in
[`training/trainer.py`](training/trainer.py). GraphNorm running stats are
left alone (no `update_bn` pass).

### Cross-dataset training pooling

For catalytic kinetics measured across multiple datasets (kcat, Km, Ki/kcat_km
when multi-dataset), `--pool_train_csvs` lets the training script append
extra `train.parquet` from sibling dataset folders. Test/val always come
from the primary `--datafolder`. See `training/rerun_12_datasets.sh` for the
exact pool maps used in the v2 rerun.

### Parameter count

v1 ≈ 25M parameters; v2 ≈ 26M (+1280×400 + 400×200 in the shortcut MLP).
Negligible memory overhead, no impact on batch size 16.

---

## 6. RF gap analysis & redesign hypothesis

PSICHIC currently loses to Random Forest by 2–8% RMSE on every catalytic
kinetic except Ki, where it ties on cold_drug and wins on random. RF takes
mean-pooled raw protein embeddings (ESM2/ESMv1/MutaPLM/ProteinCLIP) plus
ligand fingerprint features.

| Kinetic | Split | PSICHIC v1 | RF best | Gap | Winner |
|---|---|---|---|---|---|
| Ki | random | 1.228 | 1.266 | −0.038 | **PSICHIC** |
| Ki | cold_drug | 1.288 | 1.379 | −0.091 | **PSICHIC** |
| Km | cold_drug | 0.939 | 0.939 | 0 | tie |
| Km | random | 0.782 | 0.766 | +0.017 | RF |
| kcat/Km | random | 1.225 | 1.211 | +0.014 | RF |
| kcat | cold_protein | 1.176 | 1.136 | +0.040 | RF |
| kcat | random | 0.876 | 0.835 | +0.041 | RF |
| Ki | cold_protein | 1.586 | 1.502 | +0.084 | RF |
| kcat/Km | cold_drug | 1.472 | 1.419 | +0.053 | RF |
| Kd | cold_protein | 1.206 | 1.153 | +0.053 | RF |
| Kd | cold_drug | 1.216 | 1.160 | +0.056 | RF |
| Km | cold_protein | 0.917 | 0.849 | +0.068 | RF |
| kcat | cold_drug | 1.131 | 1.060 | +0.071 | RF |
| kcat/Km | cold_protein | 1.426 | 1.346 | +0.080 | RF |
| Kd | random | 1.022 | 0.935 | +0.087 | RF |

### Three structural reasons RF wins

1. **Information bottleneck.** PSICHIC projects the 1280-dim embedding to
   200 via `prot_evo` *before* fusion with AA features. For Ki (precise
   binding geometry) the GNN adds enough value to overcome the bottleneck;
   for kcat / Km / Kd the raw embedding mean already carries most of the
   signal and the bottleneck destroys it.
2. **Self-derived graph structure compounds noise.** Contact-map edges are
   sigmoid-thresholded inner products of the same ESM embeddings used as
   node features. When ESM is noisy, you double-count the noise. RF is
   immune.
3. **MinCut auxiliary losses misalign the optimization.** The Tier 1
   ablation confirmed this: dropout / LR / early-stop tweaks did not help
   because the actual misalignment is between the clustering objective
   and kinetic prediction.

### Redesign hypothesis

Add an embedding shortcut + drop MinCut + add SWA. The shortcut directly
addresses (1); attention pooling drops (3); SWA gives RF-like ensembling
for free. The graph branch is preserved so the Ki-win signal stays.

### Promotion gate

The redesign promotes to paper figures only if the v2 rerun shows:

* Catalytic kinetics gate: v2 ≤ RF best on at least 3 of {kcat, Km, Kd,
  kcat/Km}, evaluated by best-2-of-3 splits per kinetic OR per-kinetic
  mean. (See `analysis/promotion_gate.py`.)
* Ki preservation guard: v2 within +0.02 RMSE of v1 on each Ki rerun.

If catalytic gate fails, iterate the redesign before generating figures.
If only Ki guard fails, ship v1 weights for Ki tasks via
`weights_dir/version_map.json`.

---

## 7. Benchmark snapshot

The cleaned best-per-(kinetic × split) table is at
[data/benchmark/tables/table_main_best_per_kinetic_split.md](data/benchmark/tables/table_main_best_per_kinetic_split.md)
(regenerated whenever `clean_benchmark.py` + `generate_tables.py` are run).
Counts after cleanup: **1,819 rows survived** out of 1,847 raw, 28 dropped
(34 missing-required-field rows, 1 NaN-RMSE row, 1 duplicate). The full
normalization log is at
[data/benchmark/normalization_log.md](data/benchmark/normalization_log.md).

---

## 8. Next steps (priority-ordered)

### Blocking — HPC side (LSF jobs)

1. **Smoke-test v2.** Submit a quick GPU job to catch import / config bugs
   before the 14-task sweep:
   ```bash
   bsub < training/run_v2_smoke_test.lsf
   ```
   Note the returned job ID (`<SMOKE_ID>`). The job exits with PASS if it
   produces a `test_prediction_seed1.csv`.
2. **Submit the v2 rerun array.** 14 array tasks chained on the smoke job:
   ```bash
   bsub -w "done(<SMOKE_ID>)" < training/run_v2_rerun_array.lsf
   ```
   Or skip the smoke test entirely if you trust the code:
   ```bash
   bsub < training/run_v2_rerun_array.lsf
   ```
3. **(Optional) Submit the v1 baseline rerun array.** Apples-to-apples v1
   reproduction on the same 14 dataset configs:
   ```bash
   bsub < training/run_v1_baseline_array.lsf
   ```

### Blocking — local (after HPC jobs finish, copy results back)

4. **Aggregate.** `python analysis/aggregate_v2_results.py
   --results_root /path/to/TRACE_Kin_Results_v2 --output data/benchmark/trace_kin_v2_results.csv`
5. **Review the improvement.** `python analysis/check_improvement.py` —
   produces a per-task table + per-kinetic verdict + headline summary.
   Output: `data/benchmark/improvement_summary.md`.
6. **Run the gate.** `python analysis/promotion_gate.py`. Read
   `data/benchmark/promotion_gate_decision.md` for the binding verdict.

### After the gate passes

7. Regenerate the cleaned CSV: `python analysis/clean_benchmark.py`.
8. Regenerate tables with v2 column populated:
   `python analysis/generate_tables.py`
9. Generate v1-vs-v2 figure: `python figures/generate_figures.py`
10. Run significance tests including v1-vs-v2:
    `python analysis/significance_tests.py`
11. Spot-check the inference API with both v1 and v2 weights against a
    small held-out set.

### Deferred (P2 / P3 from the TRACE plan)

12. Multi-task joint head (Km + kcat + kcat/Km in one model).
13. Embedding ensemble at inference time (ESM2 + MutaPLM with
    `dual_embedding_confidence`).
14. Methods-section architecture write-up — this document's §4 / §5 is the
    starting draft.

### Reviewing improvement quickly

After step 4, the one-command answer to "did v2 actually improve over v1
and close the RF gap?" is:

```bash
python analysis/check_improvement.py
```

It prints (and writes to `data/benchmark/improvement_summary.md`) a per-task
table with `v1_rmse / v2_rmse / rf_best / Δv1 / Δrf / verdict` columns, a
per-kinetic aggregate with PASS/FAIL per kinetic, and a one-paragraph
headline you can paste into a status update. Exit code 0 if the catalytic
gate passes, 1 otherwise — same logic as `promotion_gate.py` but with
the full numeric breakdown.

---

## 9. Reproducing benchmark runs on HPC

The recommended pattern is the LSF job arrays in `training/`. They are pure
`#BSUB`-driven LSF files; submit each one with `bsub < <file>.lsf`.

### v2 rerun (the redesign)

Smoke test → rerun array, chained:

```bash
cd /sc/arion/projects/.../TRACE_Kin
bsub < training/run_v2_smoke_test.lsf            # note the job ID
bsub -w "done(<SMOKE_ID>)" < training/run_v2_rerun_array.lsf
```

The array LSF (`#BSUB -J trace_kin_v2[1-14]`) spawns 14 GPU tasks, each
running one of the 14 (dataset, kinetic, split, embedding, pool) tuples
with the v2 redesign + SWA + cross-dataset pooling where applicable.
Walltime cap per task: 48h. Defaults are appropriate for the Mount Sinai
H100 NVL queue; override `ENZYME_BASE`, `RESULT_BASE`, `CONFIG_PATH`,
`EPOCHS`, `BATCH_SIZE`, `SEED`, `LRATE` via env if needed.

### v1 baseline rerun (apples-to-apples comparator)

```bash
bsub < training/run_v1_baseline_array.lsf
```

Same 14 configs, `model_version=v1`, no SWA, no pooling. Lands in
`TRACE_Kin_Results_v1_repro/` so it doesn't collide with the historical
v1 results.

### Single-dataset reruns

The pre-existing `training/run_benchmark.lsf` still works for one-off
single-dataset training (with `DATASET_SOURCE` + `SINGLE_DATASET` env
vars). It is more flexible than the arrays but less convenient for
sweeps.

```bash
DATASET_SOURCE=MPEK_dataset \
SINGLE_DATASET=MPEK_kcat_ESMv1_embedding_random \
RESULT_BASE=/sc/arion/projects/.../TRACE_Kin_Results_v2 \
SKIP_COMPLETED=false \
MODEL_VERSION=v2 \
USE_SWA=true \
bsub < training/run_benchmark.lsf
```

### Legacy bash wrapper (fallback)

`training/rerun_12_datasets.sh` loops `bsub` over the 14 v2 jobs. It
remains in the repo but the LSF arrays above are the **recommended**
entry points — they match the user's `bsub < file.lsf` workflow and let
you chain via `-w "done(...)"` cleanly.

---

## 10. Local development

### Environment

The original PSICHIC envs in `environment_*.yml` work for the new code too —
`models/trace_kin/` reuses the same `torch_geometric`, `rdkit`, `fair-esm`
stack. For laptop development of the analysis / figures scripts only:

```bash
pip install pandas numpy scipy matplotlib scikit-learn lifelines
# torch + pyg not required for analysis/, figures/, or inference.confidence
```

### Regenerate analysis deliverables

```bash
python analysis/clean_benchmark.py
python analysis/generate_tables.py
python analysis/significance_tests.py
python figures/generate_figures.py
```

### Smoke test the model code

```bash
python -m py_compile models/trace_kin/net_v1.py models/trace_kin/net_v2.py training/trainer.py training/train_trace_kin.py
```

---

## 11. Confidence metric design notes

`inference/confidence.py` implements
`dual_embedding_confidence(pred_a, pred_b, ...)` per the
[research design doc](trace_doc/TRACE_research_design.md) Claim 3:
embedding disagreement is a useful uncertainty signal. The score is
`sigmoid(scale_midpoint − |Δ| / rmse_k)` where `rmse_k` is the historical
benchmark RMSE for the kinetic (the natural scale: divergence larger than
typical model error is a real divergence). At equal predictions the score
is `sigmoid(scale_midpoint)` ≈ 0.73 (default `scale_midpoint=1.0`); at
`Δ = rmse_k` the score is 0.5; at large divergence it asymptotes to 0.

For ensemble use, call the predictor twice (ESM2 and MutaPLM weights) on
the same SMILES/sequence and pass both outputs to
`dual_embedding_confidence`. Low confidence → flag the prediction for the
reasoning module as mutation-sensitive.

---

## 12. Reverse map: where each old document went

| Old file | Where it lives now |
|---|---|
| `HPC_TRAINING_GUIDE.md` | §9 (HPC bsub recipes) + `to_remove/legacy_docs/` |
| `TRAINING_GUIDE.md` | §10 (local development) + `to_remove/legacy_docs/` |
| `PSICHIC_Training_Analysis.md` §1.5–§1.8 | §4 (v1 architecture) + `to_remove/legacy_docs/` |
| `PSICHIC_Training_Analysis.md` §3 | §6 (RF gap analysis) + `to_remove/legacy_docs/` |
| `PSICHIC_Kin_Next_Phase_Decision.md` | §2 (status — Tier 1 halted) + `to_remove/legacy_docs/` |
| `Tier1_Rerun_Jobs.md` | §6 (gap analysis table — same source numbers) + `to_remove/legacy_docs/` |
