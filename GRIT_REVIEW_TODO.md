# GRIT implementation: points to fix

Source: line-by-line read of `models/trace_kin/grit.py` against the authors' code
(LiamMa/GRIT: `grit_layer.py`, `grit_model.py`, `rrwp_encoder.py`, `rrwp.py`,
`zinc-GRIT-RRWP.yaml`). Read-only review: nothing was executed (no torch locally).
Reviewed against repo state `origin/main` @ `79024ed`.

## Status (2026-11-03, branch `development`)

Applied in `models/trace_kin/grit.py`: items 1, 2, 4, 5 and the memory risk. Item 3 (initialisation) is
implemented as an option, `mol_grit_ref_init` (default `false`), because its effect on training should be
measured, not assumed. Checked by `tests/test_grit_fidelity.py` (needs torch + torch_geometric):
RRWP vs a transcription of the reference `add_full_rrwp`, the layer vs a transcription of the reference
`GritTransformerLayer` with its own parameter layout, init statistics vs `xavier_normal_`, and a
forward/backward smoke test. Not yet measured: the effect on RMSE of any of these changes.
Any GRIT checkpoint trained before this change used the old orientation and pair set and is not
comparable with new runs.

## Changes that can alter results

- [x] **1. Pairs are sparse, the reference attends over all pairs** (`atom_rrwp`, grit.py:86-89)
  - Reference: `full_attn: True` pads every pair inside a molecule with a zero RRWP vector.
  - Ours: keeps only pairs with nonzero RRWP within 20 hops.
  - Same result only for connected molecules whose longest shortest path is <= 20 bonds.
  - Differs for multi-fragment ligands (salts, counterions) and long molecules
    (CoA, NAD(P)H, ATP, lipids, peptides): no attention across fragments or beyond 20 hops.
  - BatchNorm on the pair features (`bn_e`) also sees a different set of rows.
  - The docstring claim "carry no signal" (grit.py:60-62) is not accurate; fix the wording either way.
  - Done: attention is now full (every pair inside a molecule; zero RRWP beyond 20 hops / across fragments).
  - First step: count affected ligands (more than one fragment, or diameter > 20) with RDKit on the HPC.

- [x] **2. RRWP orientation is transposed vs the reference** (`atom_rrwp`, grit.py:86-89; layer src/dst, grit.py:171)
  - Reference stores P^k[i,j] on edge (src=j, dst=i), i.e. P^k[dst,src]; it swaps row and col on purpose.
  - Ours stores P^k[src,dst].
  - Row-normalised P, so the two differ by a degree ratio.
  - Fix: swap the row/col of `pair_index` when building `pair_attr`. Bond embedding is symmetric, so unaffected.
  - Done: `atom_rrwp` now returns `pair_index = (src=j, dst=i)` for the value `[(D^-1 A)^k]_ij`.

- [~] **3. Initialisation differs from the reference** (grit.py:130-148)
  - Reference: `xavier_normal_` on Q, K, V, E, `A_w`. Ours: PyTorch default `Linear` init and `A_w ~ N(0, 1/dh)`.
  - Degree scaler: reference coefficients are random (std about 0.12 at width 64); ours start at 1 and 0.
  - Affects early training, not what the model can represent. Optional.
  - Done as an option: `mol_grit_ref_init: true` in the config; default false. Still to do: a short
    sweep with both inits before choosing.

- [x] **4. Extra biases** (grit.py:131-132, 280)
  - K and V have biases in ours; the reference has none.
  - The pair-PE linear has a bias in ours; the reference uses `bias=False`.
  - Node-PE linear (19 channels + bias) is equivalent to the reference (21 channels, constant identity channel): no action.

- [x] **5. Dropout placement** (grit.py:206-210)
  - Ours adds a dropout on the FFN output; the reference drops out the attention output before the degree scaler.
  - Inert while `mol_grit_dropout` is 0.0; would differ if raised.

## Deliberate adaptations: document, don't "fix"

- [ ] Atom input = type embedding + MLP over atom features + LayerNorm (reference: type embedding only). Bond types: 5 vs 4.
- [ ] No graph-level head; `atom_h` is lifted to `d_model` for pocket attention / MMCAF.
- [ ] `config_v7_grit.json` is not the paper operating point:
  - `mol_grit_dim` 512 and `mol_grit_layers` 5 (reference: 64 and 10)
  - `lrate` 1e-3, `weight_decay` 0.005466 (reference: 1e-5; effective decay about 550x larger)
  - `warmup_iters` 10000
  - Do not label results "GRIT (paper config)".
- [ ] Dead keys in `config_v7_grit.json`: `mol_pe_mode`, `mol_pe_steps`, `mol_pe_dim` are unused in the GRIT path (issue #12, point 4).

## Practical risk

- [x] **Memory:** `atom_rrwp` builds a dense N x N x 21 tensor over all atoms in the batch on every forward pass
  (the reference precomputes once per molecule). Memory is quadratic in atoms per batch.
  Fine at batch size 16; a large eval batch or a large ligand could run out of memory.
  Option: compute per molecule, or cache.
  Done: RRWP is built on a padded `(B, n_max, n_max)` tensor. 128 molecules x 50 atoms: 27 MB, 25 ms on CPU
  (the old design needed about 3.4 GB for the same batch). Numerically identical per molecule.

## Verified to match the reference (no action)

Attention order (Q+K, x E_w, signed sqrt, + E_b, ReLU), clamp +/-5, per-destination softmax, attention dropout,
value path + per-head `VeRow`, edge update (`O_e`, residual, BN), degree-scaler form, `O_h`, BN1 / FFN (2x, ReLU) / BN2,
D^-1 A row-normalisation, 21 channels [I, A, ..., A^20], self-pairs kept, bond embedding added to the pair embedding,
8 heads, edge width = hidden width, `mol_pe_raw_norm: "none"` (fixed in `79024ed`).

## Suggested verification

- [ ] On the HPC (partly covered by `tests/test_grit_fidelity.py`): load the reference `GritTransformerLayer` with identical weights and compare layer outputs
      (confirms the "matches" list above, and measures the effect of items 1 and 2).
