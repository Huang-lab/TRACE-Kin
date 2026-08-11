# 3. Methodology

## 3.1. Preliminaries

**Problem Definition.** The task of predicting enzyme kinetic parameters can be
formulated as a conditional regression problem. Given an enzyme $E$ with amino acid
sequence $\mathcal{S}_e=\{s_i\}_{i=1,\dots,L_e}$, a substrate molecule $\mathcal{S}_m$
(SMILES), and a residue contact graph $\mathcal{G}_p$ derived from the sequence, the goal
is to predict quantitative kinetic parameters
$\hat{y}=\{k_{\text{cat}},K_m,K_i,K_d\}$. Formally, we learn a multimodal regression
function $f_\theta(\cdot)$ parameterized by $\theta$:

$$
\hat{y}=f_\theta(\mathcal{S}_e,\mathcal{S}_m,\mathcal{G}_p),\qquad \hat{y}\in\mathbb{R}.
\tag{1}
$$

A pretrained protein language model (PLM) maps $\mathcal{S}_e$ to per-residue embeddings
$\mathbf{E}\in\mathbb{R}^{L_e\times C}$, where $C$ is the PLM width. We write $D$ for the
model latent dimension and $\oplus$ for concatenation.

**Existing Formulation.** Most existing methods first project the PLM embedding to a
low-dimensional space and only then aggregate over residues, approximating $f_\theta$ as

$$
\hat{y}=\psi\Big(\mathrm{Pool}\big(\phi(\mathbf{E})\big)\;\oplus\;\mathrm{Pool}\big(\mathbf{H}_m\big)\Big),
\qquad \phi:\mathbb{R}^{C}\!\rightarrow\!\mathbb{R}^{D'},\; D'\!\ll\!C,
\tag{2}
$$

where $\phi$ is a per-residue projection, $\mathrm{Pool}(\cdot)$ a permutation-invariant
reduction, and $\psi(\cdot)$ a regression head. This *compress-then-pool* ordering is
convenient but lossy: the projection is applied before any residue is weighted, so
information is discarded uniformly across the sequence irrespective of catalytic
relevance. Empirically, a Random Forest over the *unprojected* mean-pooled embedding
remains competitive with, and on catalytic endpoints superior to, graph models built on
Eq. (2) — indicating that the bottleneck, not the absence of structural modeling, limits
these pipelines.

**Readout-Centric Formulation.** We instead treat structural computation and embedding
aggregation as **distinct roles**: structure determines *where* on the enzyme to read,
while the PLM embedding supplies *what* is read, at its native width. Writing
$\Phi(\cdot)$ for a structural localizer producing a per-residue relevance score
$\mathbf{s}\in\mathbb{R}^{L_e}$ and $\mathcal{R}(\cdot)$ for a score-conditioned readout,
we reformulate the prediction function as

$$
\hat{y}=\psi\Big(
\underbrace{\Phi(\mathcal{S}_e,\mathcal{G}_p,\mathcal{S}_m)}_{\text{(i) pocket localization}}
\;\longrightarrow\;
\underbrace{\mathcal{R}\big(\mathbf{E},\,\mathbf{s}\big)}_{\text{(ii) conditioned readout}}
\Big).
\tag{3}
$$

To realize this decomposition we introduce **Structure-Guided Embedding Distillation
(SGED)**, comprising (i) a graph-transformer **structure stream** that localizes the
substrate-proximal pocket, and (ii) a **Pocket-Conditioned Embedding Readout (PCER)** that
aggregates the uncompressed embedding under the resulting weights. The two streams share
no parameters and are coupled only through $\mathbf{s}$, which is detached, so aggregation
order is inverted relative to Eq. (2): *pool at width $C$, compress afterwards*.

## 3.2. Structure-Guided Pocket Localization

Catalysis is spatially local: turnover is governed by a small set of active-site residues,
while the majority of the chain contributes scaffold rather than chemistry. The structure
stream therefore has one objective — identify those residues — and is deliberately given
no responsibility for carrying embedding content.

Residue states are initialized by projecting the PLM embedding to $D$ and adding a
random-walk structural encoding of the contact graph. With $\mathbf{u}^{(0)}=\mathbf{1}$
and $\deg(j)$ the degree of residue $j$, the $T$-step diffusion profile is

$$
u^{(k)}_i=\!\!\sum_{j\in\mathcal{N}(i)}\!\frac{u^{(k-1)}_j}{\deg(j)},\quad
\mathbf{H}^{(0)}=\mathrm{LN}\big(\mathbf{E}\mathbf{W}_p\big)
+\big[\mathbf{u}^{(0)}\oplus\cdots\oplus\mathbf{u}^{(T-1)}\big]\mathbf{W}_{\text{pe}}.
\tag{4}
$$

The graph $\mathcal{G}_p$ is obtained by thresholding a predicted contact map
$\mathbf{P}\in[0,1]^{L_e\times L_e}$ at $\tau$ and adding sequential edges
$|i-j|\le2$ to guarantee connectivity, with edge weights $w_{ij}=\mathbf{P}_{ij}$.

Contact graphs are sparse and locally reliable but cannot express allosteric coupling
between distant residues. We therefore encode residues with $\ell$ hybrid layers that
combine local message passing over $\mathcal{G}_p$ with global attention over the chain.
The local path is a GATv2 convolution with $H_\ell$ heads and radial-basis edge features
$\mathbf{e}_{ij}$,

$$
\mathbf{h}^{\text{loc}}_i=\Big\|_{k=1}^{H_\ell}\!\!\sum_{j\in\mathcal{N}(i)}\!\alpha^{k}_{ij}\mathbf{W}^{k}\mathbf{h}_j,\quad
\alpha^{k}_{ij}=\underset{j\in\mathcal{N}(i)}{\mathrm{Softmax}}\;
\mathbf{a}^{\!\top}\!\mathrm{LeakyReLU}\big(\mathbf{W}[\mathbf{h}_i\oplus\mathbf{h}_j\oplus\mathbf{e}_{ij}]\big),
\tag{5}
$$

and the global path is masked multi-head self-attention with $H_g$ heads. The two are
fused additively and followed by a feed-forward block, both residual:

$$
\mathbf{H}'=\mathrm{LN}\big(\mathbf{H}+\mathbf{h}^{\text{loc}}+\mathbf{H}^{\text{glob}}\big),\qquad
\mathbf{H}''=\mathrm{LN}\big(\mathbf{H}'+\mathrm{FFN}(\mathbf{H}')\big).
\tag{6}
$$

Substrate-conditioned relevance is then obtained by cross-attention with residues as
queries and substrate atoms as keys. Substrate atoms are encoded by a PNA message-passing
network over the molecular graph, augmented by a chemical language model embedding
projected to $D$, yielding $\mathbf{H}_m\in\mathbb{R}^{L_m\times D}$. With
$\mathbf{A}_{em}\in\mathbb{R}^{H\times L_e\times L_m}$ the multi-head attention tensor,

$$
\mathbf{A}_{em}=\mathrm{Softmax}\!\left(\frac{(\mathbf{H}''\mathbf{W}_Q)(\mathbf{H}_m\mathbf{W}_K)^{\!\top}}{\sqrt{d_k}}\right),
\qquad
s_i=\frac{1}{H L_m}\sum_{h=1}^{H}\sum_{a=1}^{L_m}\big[\mathbf{A}_{em}\big]_{h,i,a}.
\tag{7}
$$

Only the attention weights are retained; the value-weighted output is discarded, so this
module acts purely as a **localizer**. The top-$K_p$ residues by $s_i$ define the pocket
index set $\mathcal{P}\subset\{1,\dots,L_e\}$.

## 3.3. Multi-Modal Cross-Attention Fusion

Recognition is bidirectional: the pocket constrains which substrate moieties are
presented to the catalytic machinery, and the substrate in turn determines which pocket
residues are engaged. We refine both sides by **Multi-Modal Cross-Attention Fusion
(MMCAF)**, applying cross-attention in each direction between the pocket rows
$\mathbf{H}''[\mathcal{P}]$ and the substrate states $\mathbf{H}_m$.

A single pocket summary would discard scaffold context that modulates stability and
expression, whereas a purely global summary dilutes the active site. We therefore pool at
both levels and interpolate with a learned gate:

$$
\mathbf{z}_{\text{struct}}=g\odot\mathrm{Pool}\big(\tilde{\mathbf{H}}[\mathcal{P}]\big)
+(1-g)\odot\mathrm{Pool}\big(\mathbf{H}''\big),\quad
g=\sigma\Big(\mathrm{MLP}\big[\mathrm{Pool}(\tilde{\mathbf{H}}[\mathcal{P}])\oplus\mathrm{Pool}(\mathbf{H}'')\big]\Big),
\tag{8}
$$

where $\mathrm{Pool}(\mathbf{X})=\sum_i\pi_i\mathbf{x}_i$ with
$\pi=\mathrm{Softmax}(\mathrm{MLP}(\mathbf{X}))$ is masked attention pooling. The substrate
summary $\mathbf{z}_{\text{mol}}$ is obtained by the same pooling over the refined
substrate states.

## 3.4. Pocket-Conditioned Embedding Readout

PCER is the component that realizes stage (ii) of Eq. (3). The PLM embedding is
layer-normalized but **never projected** prior to aggregation, so no information is
discarded before residues have been weighted.

Since PLM pretraining objectives encode mutation sensitivity at residue level, we first
modulate the localization score by a gate learned from the raw embedding, then detach it:

$$
\gamma_i=\sigma\big(\mathrm{MLP}(\tilde{\mathbf{e}}_i)\big),\qquad
\tilde{s}_i=\mathrm{sg}\big[s_i\big]\cdot\gamma_i,
\tag{9}
$$

where $\mathrm{sg}[\cdot]$ denotes stop-gradient. Detaching $s_i$ confines the structural
parameters to the objective of Eq. (8): the readout consumes the localization but cannot
reshape it, preventing the high-capacity embedding channel from dominating pocket
identification during joint training.

Catalytic determinants act at several spatial extents — a handful of catalytic residues,
a wider binding pocket, and whole-chain properties. Rather than commit to one, we read out
at $R=3$ scales with index sets $\mathcal{S}_r$ containing the top-$K_r$ residues
($K_1\!\ll\!K_2\!<\!K_3=L_e$):

$$
w^{(r)}_i=\frac{\exp(\tilde{s}_i)\,\mathbb{1}[i\in\mathcal{S}_r]}{\sum_{j\in\mathcal{S}_r}\exp(\tilde{s}_j)},
\qquad
\mathbf{p}^{(r)}=\sum_{i=1}^{L_e}w^{(r)}_i\,\tilde{\mathbf{e}}_i\;\in\;\mathbb{R}^{C}.
\tag{10}
$$

Each pooled vector is compressed by its own network and the scales are merged by a learned
gate, so the operative extent is selected per sample and is directly inspectable:

$$
\mathbf{f}^{(r)}=\mathrm{MLP}_r\big(\mathbf{p}^{(r)}\big),\quad
\boldsymbol{\beta}=\mathrm{Softmax}\Big(\mathrm{MLP}_\beta\big[\mathbf{f}^{(1)}\oplus\mathbf{f}^{(2)}\oplus\mathbf{f}^{(3)}\big]\Big),\quad
\mathbf{z}_{\text{pcer}}=\sum_{r=1}^{R}\beta_r\mathbf{f}^{(r)}.
\tag{11}
$$

Aggregation in Eq. (10) occurs at width $C$ and compression follows in Eq. (11) — the
inverse of Eq. (2), and the central methodological change of this work.

To prevent the readout from collapsing onto a narrow subspace of the PLM manifold, we add
a reconstruction term requiring $\mathbf{z}_{\text{pcer}}$ to retain the sequence-level
embedding, with $\bar{\mathbf{e}}$ the masked mean of the normalized per-residue
embeddings:

$$
\mathcal{L}_{\text{PCER}}=\big\|\mathbf{W}_{\text{dec}}\mathbf{z}_{\text{pcer}}-\mathrm{sg}[\bar{\mathbf{e}}]\big\|_2^2 .
\tag{12}
$$

Unlike clustering-based auxiliary objectives, which impose a partitioning criterion
independent of the regression target, this term constrains only that the readout remain
informative and is therefore aligned with the primary task.

## 3.5. Kinetic Regression and Optimization

The interaction fingerprint concatenates the substrate, structural, and embedding
summaries,
$\mathbf{z}=\mathbf{z}_{\text{mol}}\oplus\mathbf{z}_{\text{struct}}\oplus\mathbf{z}_{\text{pcer}}\in\mathbb{R}^{3D}$,
and is decoded by $M$ parallel heads whose outputs are averaged to reduce head-initialization variance:

$$
\hat{y}=\frac{1}{M}\sum_{m=1}^{M}\mathrm{MLP}_m(\mathbf{z}).
\tag{13}
$$

Kinetic constants are positive and subject to multiplicative noise, so all targets are
regressed in $\log_{10}$ space with squared error:

$$
\mathcal{L}_{\text{task}}=\big(\hat{y}-\log_{10}y\big)^{2}.
\tag{14}
$$

The full objective aggregates the task loss and the readout-preservation term through a
hyperparameter $\lambda$:

$$
\mathcal{L}=\mathcal{L}_{\text{task}}+\lambda\,\mathcal{L}_{\text{PCER}}.
\tag{15}
$$

# 4. Experiments

## 4.1. Experimental Setup

**Datasets.** We evaluate on kinetic parameter prediction from curated enzyme–substrate
measurements, each entry comprising an enzyme sequence, a substrate SMILES string, and an
experimentally measured constant. Targets are $\log_{10}$-transformed. Residue contact
graphs, substrate graphs, and PLM embeddings are precomputed once and cached, so ensemble
members share preprocessing.

**Splits.** Random splits are complemented by identity-clustered splits in which no
sequence cluster spans train and test, isolating generalization to unseen enzymes; an
analogous grouping over substrates isolates generalization to unseen chemistry. This
follows the observation that random partitioning of enzyme–substrate tables leaks close
homologs between folds and inflates apparent performance.

**Evaluation Metrics.** We report root mean square error (RMSE), mean absolute error
(MAE), Pearson correlation coefficient (PCC), and Spearman correlation, all computed on
$\log_{10}$-transformed targets. Lower RMSE and MAE and higher PCC indicate better
performance. Following practice for tree-ensemble baselines, which obtain variance
reduction from bagging, we train $K$ independently seeded models differing in
initialization and data ordering, average their predictions, and retain the standard
deviation across members as an epistemic uncertainty estimate; calibration is assessed by
the rank correlation between that deviation and absolute error.

**Implementation Details.** The structure stream uses a latent dimension $D=512$ with
$\ell=3$ hybrid layers, $H_\ell=8$ local and $H_g=8$ global attention heads, radial-basis
edge features of dimension 16, and a diffusion profile of $T=16$ steps. Pocket size is
$K_p=64$; PCER scales are $K_1=16$, $K_2=64$, and $K_3=L_e$. The prediction head uses
$M=3$ members. Contact maps are thresholded at $\tau=0.5$. Layers are gradient-checkpointed
and global attention is computed in query blocks to bound activation memory. Training uses
AdamW with learning rate $3\times10^{-4}$, 500 warmup steps and cosine decay to
$1\times10^{-6}$, weight decay $0.01$, gradient-norm clipping at $1.0$, dropout $0.1$, and
batch size 32, for at most 100 epochs with early stopping on validation RMSE (patience 25).
In the full objective $\mathcal{L}$, the hyperparameter is set to $\lambda=0.1$.
Experiments are implemented in PyTorch and PyTorch Geometric on a single NVIDIA H200.
