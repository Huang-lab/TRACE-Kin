# TRACE: Implementation Plan

**TRACE** — Transformative Reasoning Across Catalytic Enzymes

## Current Results, Task Organization, and Development Roadmap

**Last Updated:** April 29, 2026
**Target Submission:** ~16 weeks from start

---

## 1. Asset Inventory — What We Have

### 1A. Kinetic Parameter Prediction (Section 1 of Paper)

**Status: Results complete, needs packaging**

**Benchmark data:**

- 1,847 experimental configurations in `combined_metrics_sort.csv`
- Coverage: 5 kinetic types × 4+ datasets × 5 embeddings × 7 models × 3 splits
- All results have RMSE, MAE, R², Pearson correlation

**Best results summary (test set):**


| Kinetic type | Best RMSE | Best R² | Best Pearson | Best config                           |
| ------------ | --------- | ------- | ------------ | ------------------------------------- |
| Km           | 0.766     | 0.654   | 0.809        | ProteinCLIP + RF on EITLEM            |
| kcat         | 0.834     | 0.705   | 0.841        | ESMv1 + RF on MPEK                    |
| Ki           | 1.225     | 0.610   | 0.786        | ProteinCLIP/ESM2 + PSICHIC on CatPred |
| Kd           | 0.935     | 0.603   | 0.777        | ESM2 + RF on in-house                 |
| kcat/Km      | 1.210     | 0.574   | 0.766        | ESM2 + RF/XGB on EITLEM               |


**MutaPLM analysis:**

- Wins 76/235 comparisons vs. ESM2
- Wins disproportionately on cold_drug splits (ΔRMSE up to -0.23)
- Strongest with Diffusion and SVR model architectures
- Confirms complementary information to ESM2 (not replacement)

**Datasets available:**

- MPEK (Km, kcat)
- EITLEM (Km, kcat, kcat/Km)
- CatPred (Km, kcat, Ki)
- In-house (Kd)

**Code location:** [existing kinetic prediction codebase — to be organized]

**TO-DO for Section 1:**

- **P0:** Clean and normalize the CSV — fix inconsistent naming (ESM1V/ESMv1/ESM1v, MutaPLM/MUTAPLM, cold_protein/cold protein, etc.)
- **P0:** Generate publication-quality benchmark tables (main text: best per kinetic type × split; supplementary: full 1,847 rows)
- **P1:** Create embedding comparison figures (radar plots, heatmaps, scatter plots)
- **P1:** Run statistical significance tests (paired t-test or Wilcoxon between embedding types on matched tasks)
- **P1:** Write TRACE-Kin architecture description — document how PSICHIC was adapted for regression
- **P2:** Implement unified TRACE-Kin inference wrapper — single API: `predict(SMILES, enzyme_seq, embedding_type) → {Km, kcat, Ki, Kd, kcat_Km, confidence}`
- **P2:** Dual-embedding confidence metric — quantify ESM2/MutaPLM agreement as uncertainty signal
- **P3:** Multi-task joint prediction experiment — train single model for Km + kcat simultaneously, compare vs. separate models

---

### 1B. Enzymatic Reaction Generation (Section 2 of Paper)

**Status: Core results complete, priority improvements identified**

**Current results (T5 baseline):**


| Split        | Forward top-1                 | Forward top-3 | Reverse top-1   |
| ------------ | ----------------------------- | ------------- | --------------- |
| Random       | 77.04%                        | 91.31%        | 66.69%          |
| Cold drug    | 39.47% (exp1) / 41.10% (exp2) | —             | 13.51% / 14.35% |
| Cold protein | 75.76% / 73.59%               | —             | 58.28% / 62.22% |


**Reranker results:**

- Fallback accuracy (correct in beam): 85.94% random, 68.77% cold drug, 85.93% cold protein
- Does NOT improve top-1 over T5 original ranking
- Provides interpretable cross-attention (atom-mapping, SOM identification)
- v2/v3 variants trained on forward/reverse, all splits

**GNN+T5 fusion results:**

- Add fusion: catastrophic failure (77% → 6%)
- Root cause documented in WHY_HYBRID_LOW.md: single-vector projection destroys T5 representations
- Gate, cross-attention, multitoken fusion scripts exist but not yet evaluated

**Code assets:**


| Directory           | Contents                                                          | Status   |
| ------------------- | ----------------------------------------------------------------- | -------- |
| `models/`           | PSICHIC backbone, T5GraphFusion, GraphReranker, hybrid model      | Complete |
| `utils/`            | Datasets, SMILES processing, graph construction, generation utils | Complete |
| `scripts/`          | Training/eval scripts, LSF job scripts                            | Complete |
| `prediction_app/`   | Inference pipeline (predict, prepare, aggregate, evaluate)        | Complete |
| `reranker_explain/` | Interpretability scripts (attention visualization, PyMol)         | Complete |
| `result/`           | All checkpoints, metrics, cancer target predictions               | Complete |
| Docs                | README, MODEL_DOCUMENTATION, PROJECT_STATUS, etc.                 | Complete |


**TO-DO for Section 2:**

- **P0 (1 day):** Score fusion for reranker — `score = α × T5_logprob + (1-α) × reranker_score`, grid search α on validation set. Eval-only, no retraining. Expected: improved top-1 without changing any model.
- **P0 (1 week):** GNN+T5 with frozen T5 encoder + cross-attention fusion — critical experiment for the paper. Freeze T5 encoder entirely, inject PSICHIC graph features via cross-attention only. This parallels BioReason-Pro's approach (ESM3 embeddings enter Qwen3 via cross-attention, never corrupting pretrained weights).
  - Script: adapt existing `scripts/` for frozen encoder mode
  - Config: cross_attention fusion only, batch=4, grad_accum=4, 50 epochs
  - Eval: all 6 splits (forward/reverse × random/cold-drug/cold-protein)
  - Success criterion: recover T5 baseline accuracy while gaining structural awareness
- **P1 (1 week):** GNN+T5 with LoRA (rank=8, alpha=16, q/v projections) — alternative if cross-attention alone is insufficient. Scripts ready for 3 forward splits.
- **P1 (3 days):** Reranker with margin loss (replace listwise CE) — expected to improve reranker standalone performance.
- **P1 (1 day):** Increase beam size for reverse tasks (5 → 10-20) — more training data for reverse reranker.
- **P2 (1 day):** Multi-sample confidence — generate N=10 beam samples per reaction, compute product frequency as confidence score (parallel to GO-GPT's sampling strategy).
- **P2 (1 week):** SELFIES for reverse prediction — solve 18% validity problem. Current reverse validity is a showstopper for drug metabolism application (predicting parent drugs from metabolites).
- **P2 (1 week):** Explainability visualization pipeline — PyMol integration for paper-ready interpretability figures showing atom-level attention on 3D enzyme-substrate structures.

**Active experiments (scripts ready, not yet run/evaluated):**

- GNN+T5 LoRA + cross-attention fusion (3 forward splits)
- Reranker v3 with score fusion + beam-10 (all 6 splits)
- GNN+T5 gate, cross_attention, multitoken fusion variants

---

### 1C. Cancer Target Predictions (Section 3 seed)

**Status: Preliminary results, needs extension**

**Available:**

- Aggregated prediction CSVs in `result/cancer_target_predictions/`
- Evaluation report
- Per-query directories removed (redundant with aggregated data)

**TO-DO:**

- **P2:** Select top 5-10 enzyme-drug pairs from cancer predictions for case study
- **P2:** Add TRACE-Kin profiling for selected pairs (WT vs. tumor-mutant)
- **P3:** Cross-reference with TCGA expression data for tumor-enriched enzymes

---

## 2. What Needs To Be Built

### 2A. Data Curation (Section 1 enrichment + Section 3 applications)

**Timeline: Weeks 2-5, parallel with foundation fixes**

- **P1: Drug metabolism dataset**
  - Source: DrugBank (~2,800 drugs with metabolic pathways)
  - Source: MetXBioDB (experimentally validated metabolite structures)
  - Source: FDA drug labels (clinical consequences of metabolites)
  - Map metabolites to ChEMBL bioactivity where available
  - Target: ~15K drug-enzyme-metabolite triplets with clinical annotation
  - Format: (parent_SMILES, enzyme_EC, enzyme_seq, metabolite_SMILES, clinical_label[active/inactive/toxic])
- **P1: Biocatalysis dataset**
  - Source: BRENDA substrate-product pairs with kinetic parameters
  - Source: Rhea reactions with atom mapping
  - Source: Literature late-stage functionalization reactions
  - Target: ~8K biocatalytic transformations relevant to drug optimization
- **P1: Context feature extraction pipeline**
  - For every reaction: reactant ADMET (ADMETlab3.0), product ADMET (predicted post-generation), enzyme active site features (AlphaFold/PDB), EC hierarchy path, KEGG pathway, ChEMBL bioactivity of products
  - Script: batch processing pipeline for all 128K training reactions + new drug data
  - Output: per-reaction feature JSON files
- **P2: Mutation-specific data**
  - Source: PharmGKB clinically annotated pharmacogenomic variants
  - Source: CYP allele nomenclature database (~300 characterized variants)
  - Source: BRENDA mutant kinetics (experimentally measured for enzyme variants)
  - Target: ~5K enzyme-mutation-kinetic triplets with clinical phenotype labels
- **P2: Temporal split design**
  - Training: reactions/drugs annotated before cutoff (DrugBank release 5.1 or similar)
  - Test: drugs approved or metabolites characterized after cutoff
  - Purpose: enables "de novo prediction" validation

---

### 2B. TRACE-Reason Model (Section 4)

**Timeline: Weeks 4-10**

**Phase 1: Reasoning trace schema and generation (Weeks 4-6)**

- **P0: Define reasoning trace schema**
  - Six blocks: enzyme_analysis, substrate_analysis, reaction_prediction, kinetic_interpretation, mutation_impact, clinical_implications
  - Each block specifies required evidence sources from upstream modules
  - Include confidence markers and uncertainty flags
  - Draft 10 exemplar traces manually for prompt engineering
- **P0: GPT-5 trace generation pipeline**
  - Template design: provide (reactant, enzyme, products, kinetics, ADMET, context) → structured trace
  - Generate ~100K traces across training set
  - Budget estimate: ~$3-5K API cost
  - Quality filter: validate SMILES validity, EC correctness, ADMET consistency in generated traces
  - Manual audit: review 500 random traces, iterate prompts until >90% pass
- **P1: Mutation-enriched traces**
  - For ~20K reactions (subset of 100K): generate variant traces
  - Same reaction with WT enzyme → trace with normal kinetics
  - Same reaction with mutant enzyme → trace explaining kinetic changes + clinical consequence
  - Requires mutation-specific data from 2A

**Phase 2: Model training (Weeks 6-10)**

- **P0: TRACE-Reason SFT**
  - Backbone: Qwen3-4B (matches BioReason-Pro for comparability)
  - Input architecture:
    - Reactant molecular embedding (MoLFormer) via cross-attention
    - Enzyme ESM2 embedding via cross-attention
    - Enzyme MutaPLM embedding via separate cross-attention head
    - TRACE-Gen predicted products (tokenized SMILES + confidence scores)
    - TRACE-Kin predicted kinetics (Km, kcat, Ki, Kd as structured text)
    - ADMET feature vector (projected via linear layer)
    - EC hierarchy + pathway context (tokenized text)
  - Training: 3-5 epochs, batch=8-16, grad_accum=4, lr=1e-5 cosine decay
  - LoRA rank=16 on Q/V projections
  - Compute: 48-72h on 4× H100 (Minerva cluster)
- **P1: TRACE-Reason RL**
  - Algorithm: GSPO (following BioReason-Pro)
  - Composite reward:
    - w1 × product_accuracy (top-1 match vs. known products)
    - w2 × kinetic_accuracy (predicted kinetics vs. ground truth)
    - w3 × attention_faithfulness (SOM claims match reranker attention)
    - w4 × reasoning_coherence (LLM judge sub-score)
    - w5 × chemical_validity (all SMILES in trace are valid)
    - w6 × hallucination_penalty (claims not supported by input evidence)
  - Compute: 24-48h on 4× H100

---

### 2C. Application Cases (Section 3)

**Timeline: Weeks 8-12, parallel with evaluation**

**Case A: CYP2D6/Tamoxifen Pharmacogenomics**

- **P1: Data collection**
  - Tamoxifen metabolic pathway from DrugBank/KEGG
  - CYP2D6 allele sequences (*1/*2/*4/*10/*17/*41) from PharmVar
  - Published Km/kcat values from BRENDA for validation
  - Clinical endoxifen plasma concentration data by genotype
- **P1: Pipeline execution**
  - Run TRACE-Gen on tamoxifen + each CYP2D6 variant
  - Run TRACE-Kin with ESM2 and MutaPLM for each variant
  - Extract reranker attention for SOM identification
  - Generate TRACE trace for each variant
  - Validate: predicted kinetic changes vs. published values, predicted metabolizer phenotype vs. clinical classification

**Case B: Enzymatic Drug Modification**

- **P2: Scaffold selection**
  - Select 3-5 drug scaffolds with known enzymatic modification literature
  - Candidate enzymes: P450-BM3 (hydroxylation), RebH (halogenation), OmtA (methylation)
  - Published regioselectivity and yield data for validation
- **P2: Pipeline execution**
  - TRACE-Gen: predict modification products for each enzyme-scaffold pair
  - TRACE-Kin: predict Km/kcat for each pair
  - Reranker: identify predicted modification sites
  - TRACE: rank modifications by kinetic feasibility × property improvement
  - Validate: predicted sites vs. published regioselectivity

**Case C: Cancer Enzyme Targets**

- **P2: Target selection from existing cancer predictions**
  - Filter cancer_target_predictions for enzyme targets with somatic mutations in TCGA
  - Select 3-5 cases with published tumor pharmacology data
  - Cross-reference with COSMIC database for mutation frequencies
- **P2: Mutation-aware analysis**
  - TRACE-Kin: WT vs. tumor-mutant enzyme kinetics
  - TRACE-Gen: product comparison WT vs. mutant
  - TRACE: clinical reasoning about tumor-selective metabolism

---

### 2D. Evaluation (Cross-section)

**Timeline: Weeks 9-12**

**Automated evaluation:**

- **P0: Kinetic prediction benchmarks** (Section 1)
  - Already complete from CSV — needs packaging into paper tables/figures
  - Additional: comparison against published DLKcat, UniKP, CatPred baselines
  - Run baseline tools on same datasets for head-to-head comparison
- **P0: Reaction generation benchmarks** (Section 2)
  - T5 baseline results already complete
  - Score fusion results (after P0 to-do above)
  - Frozen-T5 cross-attention results (after P0 to-do above)
  - Comparison against published ReactionT5v2, RetroTRAE baselines
- **P1: TRACE-Reason automated eval** (Section 4)
  - LLM judge scoring (1-10) across five axes: mechanism, pharmacology, mutation reasoning, specificity, reliability
  - Chemical validity rate of all SMILES in reasoning traces
  - Kinetic value accuracy in traces vs. ground truth
- **P1: Ablation studies**
  - Reasoning quality WITH vs. WITHOUT graph grounding (reranker attention input)
  - Reasoning quality WITH vs. WITHOUT kinetic input
  - Reasoning quality WITH vs. WITHOUT dual embedding (MutaPLM)
  - Each ablation: retrain SFT with input channel removed, compare LLM judge scores

**Human expert evaluation:**

- **P1: Panel recruitment**
  - Target: 15-20 medicinal chemists + DMPK scientists
  - Mount Sinai DMPK department + external collaborators
  - Design: blinded comparison, TRACE-Reason traces vs. DrugBank/FDA annotations
  - Per-axis scoring: mechanism, pharmacology, clinical utility, hallucination, overall
  - IRB requirements: check if needed for expert panel study
- **P2: Evaluation execution**
  - Select 50-100 test cases spanning diverse enzyme families
  - Generate TRACE-Reason traces for each
  - Prepare blinded comparison materials
  - Collect and analyze expert ratings
  - Calculate inter-rater reliability

**De novo validation:**

- **P2: Temporal holdout validation**
  - Select drugs approved after training cutoff
  - Run full pipeline, compare predictions vs. published post-market metabolite data
  - Target: "TRACE correctly predicted [specific metabolite] as [active/toxic]"

---

## 3. Development Phases and Timeline

### Phase 1: Foundation (Weeks 1-3)

**Goal: Solidify existing results, run critical experiments**


| Task                                         | Priority | Effort | Owner | Dependency               |
| -------------------------------------------- | -------- | ------ | ----- | ------------------------ |
| Clean kinetic CSV, generate benchmark tables | P0       | 2 days | Peter | None                     |
| Score fusion for reranker (eval-only)        | P0       | 1 day  | Peter | None                     |
| GNN+T5 frozen encoder + cross-attention      | P0       | 1 week | Peter | None                     |
| Embedding comparison figures                 | P1       | 2 days | Peter | Clean CSV                |
| GNN+T5 LoRA experiment                       | P1       | 1 week | Peter | After cross-attn results |
| Reranker margin loss experiment              | P1       | 3 days | Peter | None                     |


**Decision point (end of Week 3):** Does frozen-T5 + cross-attention work?

- YES → use as TRACE-Gen in paper
- NO → use T5 baseline + reranker as the generation story, frame GNN fusion failure as architectural insight

### Phase 2: Data + Traces (Weeks 2-6, overlaps with Phase 1)

**Goal: Build the data foundation for TRACE**


| Task                                                    | Priority | Effort | Owner | Dependency            |
| ------------------------------------------------------- | -------- | ------ | ----- | --------------------- |
| Drug metabolism dataset curation (DrugBank + MetXBioDB) | P1       | 1 week | Peter | None                  |
| Context feature extraction pipeline                     | P1       | 1 week | Peter | Drug metabolism data  |
| Mutation-specific data (PharmGKB + CYP alleles)         | P2       | 3 days | Peter | None                  |
| Reasoning trace schema design                           | P0       | 2 days | Peter | None                  |
| GPT-5 trace generation (100K)                           | P0       | 1 week | Peter | Schema + context data |
| Trace quality audit (500 samples)                       | P0       | 3 days | Peter | Generated traces      |
| SELFIES reverse prediction experiment                   | P2       | 1 week | Peter | None                  |


### Phase 3: TRACE-Reason Training (Weeks 6-10)

**Goal: Train and optimize the reasoning model**


| Task                                           | Priority | Effort         | Owner | Dependency         |
| ---------------------------------------------- | -------- | -------------- | ----- | ------------------ |
| Multimodal input architecture implementation   | P0       | 1 week         | Peter | Schema, features   |
| TRACE-Reason SFT training (Qwen3-4B)           | P0       | 3 days compute | Peter | Input arch, traces |
| SFT evaluation (LLM judge)                     | P0       | 2 days         | Peter | SFT model          |
| TRACE-Reason RL reward function implementation | P1       | 3 days         | Peter | SFT eval results   |
| TRACE-Reason RL training (GSPO)                | P1       | 2 days compute | Peter | Reward function    |
| RL evaluation                                  | P1       | 2 days         | Peter | RL model           |


**Decision point (end of Week 8):** Does SFT model produce acceptable traces?

- YES → proceed with RL optimization
- NO → iterate on trace generation quality, adjust input architecture

### Phase 4: Applications + Evaluation (Weeks 8-12)

**Goal: Execute case studies, run comprehensive evaluation**


| Task                                           | Priority | Effort         | Owner                 | Dependency                   |
| ---------------------------------------------- | -------- | -------------- | --------------------- | ---------------------------- |
| Case A: Tamoxifen/CYP2D6 data + pipeline       | P1       | 1 week         | Peter                 | Kinetic + generation modules |
| Case B: Enzymatic modification data + pipeline | P2       | 1 week         | Peter                 | Same                         |
| Case C: Cancer targets + mutation analysis     | P2       | 1 week         | Peter                 | Same + cancer predictions    |
| Expert panel recruitment                       | P1       | 2 weeks        | Peter + collaborators | None (start early)           |
| Ablation experiments (3 ablations × SFT)       | P1       | 1 week compute | Peter                 | SFT model                    |
| Expert evaluation execution                    | P2       | 2 weeks        | Peter + panel         | Recruitment, final model     |
| Temporal holdout validation                    | P2       | 3 days         | Peter                 | Temporal split               |


### Phase 5: Paper Writing (Weeks 12-16)


| Task                              | Priority | Effort | Owner | Dependency               |
| --------------------------------- | -------- | ------ | ----- | ------------------------ |
| Introduction draft                | P0       | 3 days | Peter | —                        |
| Results Section 1 (kinetics)      | P0       | 3 days | Peter | Phase 1 complete         |
| Results Section 2 (generation)    | P0       | 3 days | Peter | Phase 1 complete         |
| Results Section 3 (applications)  | P0       | 3 days | Peter | Phase 4 complete         |
| Results Section 4 (reasoning)     | P0       | 3 days | Peter | Phase 3-4 complete       |
| Methods                           | P0       | 3 days | Peter | All experiments complete |
| Figures (8 main + supplementary)  | P0       | 1 week | Peter | All results              |
| Abstract + discussion             | P0       | 2 days | Peter | Results complete         |
| Internal review + revision        | P1       | 1 week | All   | Draft complete           |
| Supplementary materials           | P1       | 3 days | Peter | Draft complete           |
| Co-author review + final revision | P1       | 1 week | All   | Internal review          |


---

## 4. Compute Budget


| Task                                     | GPUs    | Estimated time | Cluster |
| ---------------------------------------- | ------- | -------------- | ------- |
| GNN+T5 frozen cross-attention (6 splits) | 1× H100 | 6 × 48-96h     | Minerva |
| GNN+T5 LoRA (3 splits)                   | 1× H100 | 3 × 48-96h     | Minerva |
| Reranker margin loss (6 splits)          | 1× H100 | 6 × 12-24h     | Minerva |
| SELFIES reverse T5                       | 1× H100 | 48h            | Minerva |
| GPT-5 trace generation (100K)            | API     | ~$3-5K         | API     |
| TRACE-Reason SFT                         | 4× H100 | 48-72h         | Minerva |
| TRACE-Reason RL                          | 4× H100 | 24-48h         | Minerva |
| Ablation SFT (3 variants)                | 4× H100 | 3 × 48h        | Minerva |


**Total GPU-hours estimate:** ~2,000-3,000 H100-hours
**Total API cost:** ~$5K

---

## 5. Risk Register


| Risk                                          | Probability | Impact | Mitigation                                             | Fallback                                                                        |
| --------------------------------------------- | ----------- | ------ | ------------------------------------------------------ | ------------------------------------------------------------------------------- |
| GNN+T5 cross-attention fails                  | Medium      | Medium | Thorough hyperparameter search, learning rate warmup   | Use T5 + reranker as-is; frame fusion failure as architectural insight          |
| Cold-drug remains at 39%                      | High        | Low    | This is a known hard problem                           | Frame honestly as limitation; show reasoning quality maintained at low accuracy |
| Reasoning traces hallucinate chemistry        | Medium      | High   | Chemical validity filter, RL hallucination penalty     | Report hallucination rates transparently; show RL improvement over SFT          |
| Expert panel too small                        | Medium      | Medium | Start recruitment in Week 1                            | Minimum 15 evaluators; supplement with additional LLM judge analysis            |
| MutaPLM improvement not significant           | Low         | Medium | Already have 76 positive cases                         | Frame as complementary (not superior); focus on cold-drug advantage             |
| Temporal holdout has no matching ground truth | Medium      | Medium | Select drugs with known post-market metabolite studies | Use synthetic holdout from within-dataset temporal split                        |
| Compute allocation insufficient               | Low         | High   | Request early, stage experiments                       | Reduce ablation scope; prioritize SFT over RL                                   |
| GPT-5 traces have systematic errors           | Medium      | Medium | 500-sample manual audit, iterate prompts               | Use GPT-5.1 or Claude Opus 4 for generation; add post-hoc chemical validation   |


---

## 6. File Organization Plan

```
bioreason-enzyme/
├── README.md                          # Project overview + quickstart
├── PAPER_PLAN.md                      # This document
├── RESEARCH_DESIGN.md                 # Research design document
│
├── data/
│   ├── kinetics/
│   │   ├── MPEK/                      # Km, kcat data
│   │   ├── EITLEM/                    # Km, kcat, kcat/Km data
│   │   ├── catpred/                   # Km, kcat, Ki data
│   │   ├── inhouse/                   # Kd data
│   │   └── combined_metrics.csv       # Cleaned, normalized benchmark CSV
│   ├── reactions/
│   │   ├── train/                     # 128K enzyme-reaction pairs
│   │   ├── test_{random,cold_drug,cold_protein}/
│   │   └── drug_metabolism/           # DrugBank + MetXBioDB curated data
│   ├── mutations/
│   │   ├── pharmgkb_variants.csv      # Clinical pharmacogenomic variants
│   │   ├── cyp_alleles.csv            # CYP nomenclature + activity
│   │   └── brenda_mutant_kinetics.csv # Experimental mutant kinetics
│   ├── context/
│   │   ├── admet_profiles/            # Per-compound ADMET predictions
│   │   ├── ec_hierarchy.json          # EC classification tree
│   │   └── kegg_pathways.json         # Enzyme pathway memberships
│   └── traces/
│       ├── schema.json                # Reasoning trace schema definition
│       ├── exemplars/                 # 10 manually crafted exemplar traces
│       ├── generated/                 # 100K GPT-5 generated traces
│       └── audited/                   # Quality-audited subset
│
├── models/
│   ├── trace_kin/                  # TRACE-Kin (adapted from PSICHIC) for kinetic regression
│   ├── trace_gen/                 # TRACE-Gen (ReactionT5) + score fusion
│   ├── graph_reranker/            # Reranker with attention extraction
│   ├── gnn_t5_fusion/             # Frozen-T5 + cross-attention experiments
│   └── trace_reason/              # TRACE-Reason: reasoning LLM (SFT + RL)
│
├── applications/
│   ├── case_a_tamoxifen/              # CYP2D6 pharmacogenomics
│   ├── case_b_biocatalysis/           # Enzymatic drug modification
│   └── case_c_cancer/                 # Cancer enzyme targeting
│
├── evaluation/
│   ├── automated/                     # LLM judge, chemical validity, accuracy metrics
│   ├── human_eval/                    # Expert panel materials + results
│   └── ablation/                      # Ablation experiment configs + results
│
├── figures/
│   ├── fig1_overview/                 # Framework overview
│   ├── fig2_kinetics/                 # Kinetic benchmark
│   ├── fig3_embeddings/               # Embedding ensemble analysis
│   ├── fig4_generation/               # Reaction generation + reranker attention
│   ├── fig5_applications/             # Drug discovery case studies
│   ├── fig6_cancer/                   # Cancer enzyme case
│   ├── fig7_reasoning/                # Reasoning trace example
│   ├── fig8_evaluation/               # Human eval + LLM judge
│   └── supplementary/                 # Extended data figures
│
├── scripts/
│   ├── data_curation/                 # DrugBank, BRENDA, PharmGKB processing
│   ├── training/                      # All training scripts (kinetic, T5, reranker, BioReason)
│   ├── evaluation/                    # Eval scripts (benchmarks, LLM judge, expert eval)
│   └── visualization/                 # Figure generation scripts
│
└── manuscript/
    ├── main.tex                       # Main manuscript
    ├── supplementary.tex              # Supplementary materials
    ├── references.bib                 # Bibliography
    └── cover_letter.tex               # Cover letter to NCS
```

---

## 7. Weekly Milestones


| Week | Primary deliverable                                                                     | Decision/checkpoint                                         |
| ---- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 1    | Clean CSV, start score fusion + frozen-T5 experiments                                   | —                                                           |
| 2    | Score fusion results, frozen-T5 training started, drug metabolism data curation started | —                                                           |
| 3    | Frozen-T5 results, embedding comparison figures                                         | **DECISION: GNN+T5 success/fail → defines Section 2 story** |
| 4    | Drug metabolism dataset v1, reasoning trace schema, start GPT-5 generation              | —                                                           |
| 5    | Context features extracted, 100K traces generated, trace audit started                  | —                                                           |
| 6    | Trace audit complete, TRACE-Reason input architecture implemented                       | **CHECKPOINT: Trace quality >90%?**                         |
| 7    | SFT training started                                                                    | —                                                           |
| 8    | SFT training complete, initial LLM judge evaluation                                     | **DECISION: Trace quality acceptable → proceed to RL**      |
| 9    | RL training, Case A (tamoxifen) data + pipeline started                                 | —                                                           |
| 10   | RL complete, Case A results, ablation experiments started                               | —                                                           |
| 11   | Case B + C pipeline execution, expert panel evaluation started                          | —                                                           |
| 12   | All application cases complete, ablation results                                        | **CHECKPOINT: All results in hand?**                        |
| 13   | Introduction + Results drafting, figures started                                        | —                                                           |
| 14   | Methods + Discussion, figures complete                                                  | —                                                           |
| 15   | Internal review, supplementary materials                                                | —                                                           |
| 16   | Final revision, submission preparation                                                  | **SUBMIT**                                                  |


---

## 8. Success Criteria for Submission

**Minimum viable paper (must-have):**

- TRACE-Kin benchmarks across 5 kinetic types with embedding comparison (Section 1)
- TRACE-Gen (ReactionT5) with score fusion reranker + attention-based SOM visualization (Section 2)
- At least 2 application cases with full pipeline demonstration (Section 3)
- TRACE-Reason SFT model with LLM judge evaluation (Section 4)
- At least one ablation (graph grounding vs. no graph grounding)
- 8 main figures + supplementary tables

**Strong paper (target):**

- All of the above, plus:
- Frozen-T5 + cross-attention results (positive or negative — both are informative)
- RL optimization showing hallucination reduction
- 3 application cases including cancer enzyme targeting
- Human expert evaluation (15+ evaluators)
- Temporal holdout de novo validation
- Full ablation suite (3 input modality ablations)
- MutaPLM vs. ESM2 statistical significance analysis

**Exceptional paper (stretch):**

- All of the above, plus:
- Multi-task kinetic prediction (Km + kcat jointly)
- SELFIES reverse prediction solving validity problem
- Enzyme engineering case study (directed evolution validation)
- Drug-drug interaction prediction from multi-enzyme reasoning

