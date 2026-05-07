# TRACE: From Molecular Graphs to Biological Reasoning for Enzymatic Drug Discovery

## Research Design Document

**Target Journal:** Nature Computational Science
**Version:** 1.0 — April 29, 2026

---

## 1. Thesis Statement

Enzymatic reactions are fundamentally graph transformations — atoms rearrange, bonds break and form, and the enzyme's active site orchestrates which transformations occur. Yet current computational models fragment this unified process into isolated prediction tasks: kinetic parameter estimators that ignore reaction outcomes, reaction generators blind to catalytic feasibility, and neither connected to the pharmacological reasoning that drives drug discovery decisions.

TRACE (Transformative Reasoning Across Catalytic Enzymes) introduces the **graph-to-reasoning** framework — a single pipeline where 2D molecular graph representations flow from atomic-level enzyme-substrate interaction modeling, through structure-aware reaction generation, to multimodal biological reasoning that explains pharmacological consequences in natural language. By maintaining graph-level representations throughout, the framework produces predictions that are not only accurate but physically grounded and interpretable at every stage.

---

## 2. Novelty Claims (Ranked by Strength)

### Claim 1: First unified framework connecting enzymatic kinetics, reaction prediction, and pharmacological reasoning

**What exists:** Isolated tools for kinetic prediction (DLKcat, UniKP, CatPred), reaction generation (ReactionT5, RetroTRAE), and drug metabolism reasoning (manual expert annotation). No system connects these.

**What we do:** A single graph-centric pipeline where kinetic predictions inform reaction feasibility, reaction products are scored by kinetic parameters, and both feed into a reasoning model that explains drug discovery consequences.

**Why it matters:** A medicinal chemist evaluating drug metabolism currently runs separate tools and mentally integrates outputs. TRACE automates this entire reasoning chain, producing the kind of integrated analysis that previously required deep domain expertise.

### Claim 2: Graph-grounded biological reasoning — physically interpretable LLM explanations

**What exists:** LLM-based scientific reasoning (BioReason-Pro, Prot2Text) generates text explanations but these are grounded in sequence embeddings and ontology terms — abstract representations with no atomic-level physical interpretation.

**What we do:** The reasoning LLM's explanations are grounded in graph neural network attention patterns that identify specific atoms involved in enzymatic transformations, specific residues mediating enzyme-substrate contacts, and specific kinetic parameters quantifying reaction feasibility. Every claim in the reasoning trace can be traced back to a graph-derived evidence source.

**Why it matters:** Graph grounding is what makes LLM reasoning trustworthy for chemistry. When the model says "oxidation occurs at C4," this claim is backed by GNN cross-attention over the reactant-enzyme graph, not hallucinated from language patterns. This directly addresses the hallucination problem identified in BioReason-Pro (SFT model: hallucination score 6.92/10).

### Claim 3: Mutation-aware enzymatic reasoning through dual protein embedding ensemble

**What exists:** MutaPLM captures mutation effects on protein function. ESM2 captures general protein structure. Neither has been combined for enzymatic reaction reasoning.

**What we do:** Dual embedding ensemble (ESM2 + MutaPLM) where embedding agreement/disagreement itself becomes an input to reasoning. When predictions diverge, the model flags mutation-sensitive predictions — directly enabling pharmacogenomic reasoning about enzyme polymorphisms (e.g., CYP2D6 variants affecting drug metabolism).

**Why it matters:** Enzyme mutations are not edge cases — CYP pharmacogenomics affects drug dosing for millions of patients. No existing reaction prediction model accounts for how enzyme variants alter products or kinetics.

### Claim 4: Comprehensive kinetic prediction across five parameter types with graph-based enzyme-substrate interaction modeling

**What exists:** DLKcat (kcat only), UniKP (Km and kcat), CatPred (Km, kcat, Ki). Typically one dataset, one embedding, limited generalization analysis.

**What we do:** TRACE-Kin (adapted from PSICHIC) predicting Km, kcat, Ki, Kd, and catalytic efficiency (kcat/Km) across four curated datasets (MPEK, EITLEM, CatPred, in-house Kd), five protein embedding types, three generalization splits (random, cold-drug, cold-protein), and seven model architectures. Comprehensive benchmark of 1,847 experimental configurations providing the most complete picture of enzymatic kinetic prediction performance to date.

**Why it matters:** The breadth of this benchmark establishes which embedding-model-split combinations work for different kinetic types — a practical resource for the field beyond our specific framework.

---

## 3. Paper Structure and Design Logic

### Overview Figure (Fig. 1): The Graph-to-Reasoning Pipeline

The central figure shows a single molecular graph representation flowing through four levels of understanding:

```
Level 1: Graph interactions      → "How does the enzyme see this substrate?"
Level 2: Graph-guided reaction   → "What transformation occurs, and where?"
Level 3: Graph-informed design   → "Which substrates should we propose?"
Level 4: Graph-grounded reasoning → "What does this mean for the patient?"
```

Each level uses the same PSICHIC-derived graph representation (TRACE-Kin for kinetics, TRACE-Gen for reaction generation, TRACE-Reason for biological reasoning) but extracts different information. This is the visual thesis: one representation, four levels of insight, unified by the graph.

### Section 1: Graph-Level Understanding — Enzyme-Substrate Interaction Kinetics

**Design rationale:** This section establishes the graph neural network as a powerful encoder of enzyme-substrate interactions. By showing strong performance across five kinetic types and multiple datasets, we build the reader's confidence that graph representations capture the physical chemistry of enzyme-substrate recognition.

**Key design decisions:**

1. **Why PSICHIC adaptation?** PSICHIC was designed for protein-small molecule interaction classification. Adapting it for regression (kinetic parameters) leverages its bilinear attention mechanism for quantitative interaction modeling. The adaptation involves: replacing classification head with regression head, adding multi-task prediction across kinetic types, and incorporating multiple protein embedding backbones.
2. **Why five kinetic types?** Km (binding), kcat (catalysis), Ki (inhibition), Kd (dissociation), and kcat/Km (catalytic efficiency) each capture a different aspect of enzyme-substrate interaction. Predicting all five from a unified model demonstrates that the graph representation captures the full spectrum of interaction physics.
3. **Why embedding ensemble?** No single embedding dominates across all conditions:
  - ESM2/ESMv1 → best for random splits (general performance)
  - MutaPLM → best for cold-drug splits with specific architectures (mutation-dependent substrate specificity)
  - ProteinCLIP → best for cold-protein generalization (cross-enzyme transfer)
   The ensemble captures complementary information. Critically, embedding disagreement is not noise — it's signal about prediction uncertainty that feeds into the reasoning module.
4. **Why three split modes?** Random (interpolation), cold-drug (unseen substrates), and cold-protein (unseen enzymes) test fundamentally different generalization capabilities. Cold-drug is the hardest and most practically important (evaluating new drug candidates against known enzymes).

**Results to highlight:**

- Km: Pearson 0.81, RMSE 0.766 (best: ProteinCLIP + Random Forest on EITLEM)
- kcat: Pearson 0.84, R² 0.705 (best: ESMv1 + Random Forest on MPEK)
- Ki: Pearson 0.79 (best: ProteinCLIP/ESM2 + PSICHIC on CatPred)
- Kd: Pearson 0.78 (best: ESM2 + Random Forest on in-house data)
- MutaPLM wins 76/235 comparisons overall, but disproportionately on cold-drug splits (ΔRMSE up to -0.23)

**Figure 2:** Kinetic prediction benchmark

- Panel A: TRACE-Kin architecture (adapted from PSICHIC)
- Panel B: Heatmap — best RMSE across kinetic type × embedding × split (5×5×3 grid)
- Panel C: Embedding complementarity — scatter plot of ESM2 vs. MutaPLM RMSE per task, colored by split type (showing MutaPLM wins concentrated in cold-drug quadrant)
- Panel D: Cold-split degradation curves showing how each embedding type degrades from random → cold-drug

### Section 2: Graph-Guided Reaction — Predicting Enzymatic Transformations

**Design rationale:** Having established that the graph captures interaction physics (Section 1), we now show it guides reaction prediction — what products form and where the enzyme acts. The graph reranker is the bridge: it uses the same graph representation to score reaction candidates and provide atom-level interpretability.

**Key design decisions:**

1. **Why T5 + graph reranker, not end-to-end GNN generation?** Direct GNN→SMILES generation has fundamental validity issues. T5 produces chemically valid sequences; the GNN reranker adds structural awareness without disrupting generation. This is a pragmatic architectural choice supported by our negative result: GNN+T5 add fusion catastrophically fails (77% → 6% accuracy), demonstrating that naively injecting graph features into pretrained sequence models destroys their representations.
2. **Why score fusion?** T5 provides linguistic confidence (log-probability of token sequence), the reranker provides structural confidence (graph-based interaction score). Combining both captures complementary quality signals — T5 knows chemistry syntax, the GNN knows interaction physics.
3. **Why is the reranker's attention the key contribution, not its accuracy?** The reranker's top-1 selection doesn't beat T5's original ranking. But this misses the point: the reranker's cross-attention over reactant-product-enzyme graphs provides site-of-metabolism (SOM) identification for free. This attention becomes the physical grounding for the reasoning module in Section 4.
4. **Connection to Section 1:** Kinetic predictions from TRACE-Kin can inform reaction feasibility — if predicted Km is very high (weak binding), the reaction is unlikely to proceed at physiological concentrations. This cross-module signal is unique to our unified framework.

**Results to highlight:**

- TRACE-Gen baseline: 77.04% forward random top-1 (91.31% top-3)
- Cold-protein generalization: 75.76% (minimal drop, enzyme sequence not the bottleneck)
- Cold-drug challenge: 39.47% (fundamental difficulty of unseen molecules)
- Reranker fallback accuracy: 85.94% (on subset where correct answer is in beam)
- Score fusion improvement: [to be measured — Priority 1 from current to-do]
- GNN+T5 frozen encoder + cross-attention: [to be measured — Priority 2]

**Figure 4:** Enzymatic reaction generation

- Panel A: TRACE-Gen architecture showing T5 generator + graph reranker pipeline
- Panel B: Product prediction accuracy across all six split conditions (forward/reverse × random/cold-drug/cold-protein)
- Panel C: Reranker cross-attention visualization — atom-level heatmap on reactant showing predicted sites of enzymatic transformation, compared against known sites of metabolism
- Panel D: Score fusion analysis (T5 logprob vs. reranker score vs. combined)

### Section 3: Graph-Informed Drug Discovery — Applications

**Design rationale:** This section demonstrates translational value by applying Sections 1 and 2 to real drug discovery problems. Each application case uses the full pipeline and shows something that isolated tools cannot achieve.

**Key design decisions:**

1. **Why application cases, not just benchmarks?** NCS values translational impact. Pure benchmarks show technical competence; application cases show that the framework enables decisions that weren't previously possible computationally.
2. **Case selection criteria:** Each case must (a) require both kinetic prediction AND reaction generation (demonstrating the value of the unified framework), (b) demonstrate mutation-aware reasoning (unique contribution), and (c) have ground truth for validation.

**Application Case A: CYP Pharmacogenomics — Tamoxifen Metabolism**

Why this case: Tamoxifen is a breast cancer drug that requires CYP2D6-mediated bioactivation to its active metabolite endoxifen. CYP2D6 is the most polymorphic drug-metabolizing enzyme (>100 known alleles). Current clinical guidelines recommend CYP2D6 genotype testing before tamoxifen prescription.

Full pipeline demonstration:

- TRACE-Gen predicts tamoxifen → N-desmethyltamoxifen → endoxifen metabolic pathway
- TRACE-Kin predicts Km/kcat for CYP2D6 wild-type and *4, *10, *17 variants
- Dual embedding: ESM2 captures general CYP2D6 structure, MutaPLM captures allele-specific binding changes
- Graph reranker attention identifies sites of metabolism on tamoxifen scaffold
- TRACE generates clinical reasoning: "CYP2D6*4 carriers produce 75% less endoxifen → consider alternative therapy or dose adjustment"

Validation: Published pharmacokinetic studies reporting endoxifen plasma concentrations by CYP2D6 genotype.

Impact: Demonstrates that a computational model can recapitulate clinical pharmacogenomic reasoning that currently requires specialist interpretation.

**Application Case B: Enzymatic Late-Stage Drug Modification**

Why this case: Late-stage enzymatic functionalization is a growing field in medicinal chemistry — using enzymes to introduce modifications at positions inaccessible to chemical synthesis. Requires both predicting what the enzyme does (Section 2) and whether it does it efficiently enough to be practical (Section 1).

Full pipeline demonstration:

- Select a drug scaffold and screen against panel of modifying enzymes (P450-BM3 variants, halogenases, methyltransferases)
- TRACE-Gen predicts modification products for each enzyme
- TRACE-Kin predicts catalytic efficiency for each enzyme-substrate pair
- Reranker attention identifies modification sites on the drug scaffold
- TRACE ranks modifications by: kinetic feasibility × product drug-likeness × property improvement
- Generates reasoning: "P450-BM3 F87A introduces C3-hydroxylation (kcat/Km = X), improving aqueous solubility 3-fold while maintaining target binding"

Validation: Published biocatalysis studies reporting regioselectivity and yields for enzymatic drug modifications.

Impact: Shows the framework as a practical tool for medicinal chemistry — predicting which enzymatic modifications are worth attempting experimentally.

**Application Case C: Cancer Enzyme Targeting**

Why this case: Tumor-enriched enzymes with somatic mutations may metabolize drugs differently in the tumor microenvironment. This connects computational prediction to precision oncology.

Full pipeline demonstration:

- From existing cancer target predictions, select enzyme targets differentially expressed in tumor vs. normal (TCGA data)
- Predict drug metabolism by WT enzyme (normal tissue) vs. tumor-mutant enzyme
- TRACE-Kin with MutaPLM captures mutation-specific kinetic changes
- TRACE generates traces explaining tumor-selective drug metabolism
- Identifies cases where mutation improves or impairs drug activation/deactivation in tumor

Validation: Published tumor pharmacology studies; comparison against clinical drug response data where enzyme genotype is reported.

Impact: Extends the framework to precision oncology — same drug, different metabolism based on tumor enzyme genotype.

**Figure 5:** Drug discovery applications

- Panel A: Pipeline overview for drug candidate evaluation
- Panel B: Tamoxifen/CYP2D6 case — metabolic pathway prediction with mutation-stratified kinetics
- Panel C: Enzymatic modification case — ranked modifications with reranker attention showing sites
- Panel D: Cancer enzyme case — WT vs. mutant kinetic profiles with clinical interpretation

### Section 4: Graph-Grounded Reasoning — From Predictions to Explanations

**Design rationale:** This section presents the reasoning LLM not as "we also added an LLM" but as the integration layer that makes the framework greater than the sum of its parts. The key innovation is graph grounding — every claim in the reasoning trace is backed by specific graph-derived evidence.

**Key design decisions:**

1. **Why graph grounding matters:** BioReason-Pro demonstrated that SFT reasoning models hallucinate (inventing enzyme identities, reversing pathway directions). Our hypothesis: grounding reasoning in physical graph evidence — atom-level attention showing where the enzyme acts, quantitative kinetic predictions showing how fast — reduces hallucination because the model has concrete evidence to reason from rather than relying on language pattern completion.
2. **Reasoning trace structure:** Six blocks, each tied to a specific computational module:
  - Enzyme analysis ← ESM2 + MutaPLM embeddings + EC classification
  - Substrate analysis ← MoLFormer embedding + ADMET predictions
  - Reaction prediction ← TRACE-Gen products + reranker attention (SOM)
  - Kinetic interpretation ← TRACE-Kin Km/kcat/Ki predictions
  - Mutation impact ← Dual-embedding divergence + MutaPLM-specific kinetics
  - Clinical implications ← Integration of all above with drug context
3. **Training data:** ~100K synthetic reasoning traces generated by GPT-5, grounded in real predictions from Modules A and B (not hypothetical). This means the training traces contain actual TRACE-Kin predictions and TRACE-Gen outputs, not fabricated numbers.
4. **RL reward design:** Composite reward covering:
  - Product accuracy (do the products mentioned in the trace match ground truth?)
  - Kinetic accuracy (are the kinetic values in the trace consistent with predictions?)
  - Attention faithfulness (do SOM claims match reranker attention patterns?)
  - Clinical relevance (LLM judge sub-score)
  - Hallucination penalty (claims not supported by any input evidence)

**Results to highlight:**

- LLM judge score across five reasoning axes (target: >7.5/10 overall)
- Human expert preference: medicinal chemist panel comparison against DrugBank/FDA annotations
- Ablation: reasoning quality with vs. without graph grounding (attention inputs)
- Ablation: reasoning quality with vs. without kinetic inputs
- Ablation: reasoning quality with vs. without dual embedding (mutation awareness)
- Hallucination rate: SFT vs. RL (expected improvement paralleling BioReason-Pro)

**Figure 7:** TRACE-Reason evaluation

- Panel A: Full reasoning trace example for the tamoxifen/CYP2D6 case — showing how each trace block maps to a specific upstream module
- Panel B: LLM judge score distribution across five axes, compared against baselines (text-only reasoning without graph grounding)
- Panel C: Human expert preference evaluation — blinded comparison design and results
- Panel D: Ablation — contribution of each input modality to reasoning quality

---

## 4. Impact Statement

### For computational biology:

TRACE establishes the graph-to-reasoning paradigm: using graph neural network representations not just for prediction but as the physical grounding for language model reasoning. This principle extends beyond enzymology to any domain where predictions should be traceable to structural evidence.

### For drug discovery:

The framework replaces a fragmented workflow (separate kinetic, reaction, and ADMET tools) with a unified pipeline that produces actionable, interpretable reasoning. For the first time, a computational model can generate the kind of integrated metabolic assessment that previously required a DMPK specialist.

### For clinical pharmacogenomics:

Mutation-aware enzymatic reasoning directly addresses the translational gap between genotype data and drug dosing decisions. By connecting enzyme polymorphisms to kinetic changes to metabolite profiles to clinical consequences, TRACE makes pharmacogenomic reasoning accessible to non-specialist prescribers.

### For the field of AI reasoning in science:

We provide evidence that graph grounding reduces hallucination in scientific reasoning models — a finding with implications beyond enzymology for any domain where LLMs are used to explain computational predictions.

---

## 5. Anticipated Reviewer Concerns and Preemptive Responses

### "This is four papers, not one"

Response: The graph representation is the unifying element. We demonstrate via ablation that removing any one module degrades the downstream reasoning quality — proving these are not independent contributions but interdependent components of a single framework. Specifically, reasoning without kinetic input misses feasibility constraints; reasoning without reaction prediction has nothing to explain; kinetic prediction without reasoning produces numbers without context.

### "The reasoning model is just GPT-5-distilled knowledge"

Response: The reasoning traces are grounded in real predictions from our upstream models, not general GPT-5 chemistry knowledge. We validate this by showing the model correctly reasons about novel enzyme-substrate pairs not in GPT-5's training data (temporal holdout). Furthermore, RL optimization with prediction-accuracy-based rewards forces the model beyond its initialization.

### "Cold-drug performance is still weak (39%)"

Response: We present this honestly as a fundamental challenge of unseen-molecule generalization. Importantly, we show that reasoning quality is maintained even when product accuracy drops — the model correctly reasons about prediction uncertainty, flags low-confidence cases, and provides appropriate caveats. This mirrors BioReason-Pro's finding that LLM scores remained 7-8.5 even at low BLAST similarity.

### "MutaPLM doesn't always outperform ESM2"

Response: This is precisely the point. MutaPLM captures complementary information — it wins specifically on cold-drug splits where mutation-dependent substrate specificity matters. The ensemble captures both general structure (ESM2) and mutation effects (MutaPLM). Embedding disagreement itself is informative signal fed to the reasoning module.

### "Why not use 3D structure directly?"

Response: 2D graphs are more robust (no structure prediction errors), faster to compute, and applicable to enzymes without experimental structures. We note that 3D structural features (from AlphaFold) can be incorporated as future work, but the 2D graph already captures the interaction physics needed for accurate kinetic and reaction prediction.

---

## 6. Comparison with Key Related Work


| Method        | Kinetics                   | Reaction           | Reasoning      | Mutation-aware     | Graph-grounded           |
| ------------- | -------------------------- | ------------------ | -------------- | ------------------ | ------------------------ |
| DLKcat        | kcat only                  | —                  | —              | —                  | —                        |
| UniKP         | Km, kcat                   | —                  | —              | —                  | —                        |
| CatPred       | Km, kcat, Ki               | —                  | —              | —                  | —                        |
| ReactionT5    | —                          | Products           | —              | —                  | —                        |
| RetroTRAE     | —                          | Retro              | —              | —                  | —                        |
| BioReason-Pro | —                          | —                  | GO terms       | —                  | Residue attention        |
| DrugGPT       | —                          | Drug gen           | —              | —                  | —                        |
| **TRACE**     | **Km, kcat, Ki, Kd, eff.** | **Products + SOM** | **Full trace** | **Dual embedding** | **Atom-level attention** |


---

## 7. Extended Data / Supplementary Plan


| Item          | Content                                                                           |
| ------------- | --------------------------------------------------------------------------------- |
| Supp. Table 1 | Full benchmark: all 1,847 experimental configurations with RMSE, MAE, R², Pearson |
| Supp. Table 2 | MutaPLM vs. ESM2 head-to-head comparison across all cold splits                   |
| Supp. Table 3 | TRACE-Gen accuracy across all six split conditions (forward/reverse × 3 splits)   |
| Supp. Table 4 | Score fusion optimization (α grid search results)                                 |
| Supp. Fig. 1  | Per-dataset kinetic prediction performance breakdown                              |
| Supp. Fig. 2  | Additional reranker attention maps for diverse enzyme families                    |
| Supp. Fig. 3  | Additional application case studies (full reasoning traces)                       |
| Supp. Fig. 4  | RL training curves and reward component analysis                                  |
| Supp. Fig. 5  | Human evaluation design, inter-rater reliability, per-axis scores                 |
| Supp. Note 1  | TRACE-Kin architecture details and hyperparameters                                |
| Supp. Note 2  | Reasoning trace generation prompts and quality filtering                          |
| Supp. Note 3  | RL reward function design and sensitivity analysis                                |
| Code/Data     | GitHub repository with all code, model weights, and benchmark datasets            |


---

## 8. Author Contribution Framework


| Contribution                                 | Lead                       |
| -------------------------------------------- | -------------------------- |
| Conceptualization and framework design       | Peter (CW)                 |
| TRACE-Kin model development and benchmarking | Peter (CW)                 |
| TRACE-Gen development and evaluation         | Peter (CW)                 |
| TRACE-Reason model training                  | Peter (CW)                 |
| Drug discovery application case design       | Peter (CW) + collaborators |
| Human expert evaluation coordination         | Mount Sinai collaborators  |
| Manuscript writing                           | Peter (CW)                 |
| Supervision                                  | PI(s)                      |


