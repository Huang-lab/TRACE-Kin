"""TRACE-Kin v2 architecture.

Three load-bearing changes vs :class:`TraceKinV1`:

1. **Embedding shortcut branch** — a parallel path that mean-pools the raw
   per-residue embedding (1280-dim) over each protein and projects it through a
   small MLP, then concatenates with the GNN output. This guarantees the model
   has access to at least the information that Random Forest uses on the same
   embeddings, eliminating the bottleneck imposed by the v1
   ``prot_evo: 1280 → 200`` projection before fusion.

2. **Attention-pooled cluster aggregation** — replaces ``dense_mincut_pool``
   and its orthogonality / clustering auxiliary losses with a simple
   softmax-weighted aggregation of residues into K cluster nodes. The cluster
   layer's logits become a soft assignment via softmax over K, and cluster
   features are the weighted sum of residue features. The K-cluster structure
   is preserved so the existing drug↔protein cross-attention layer is
   unchanged. Auxiliary losses become 0 tensors.

3. **Embedding-dominant residue fusion** — replaces the v1 additive fusion
   ``prot_aa(x) + prot_evo(x)`` with ``prot_evo(x) + 0.1 * prot_aa(x)``. The
   fixed 0.1 weight ensures the embedding signal dominates (matching what RF
   uses) while AA features remain available as a small physicochemical
   correction.

Forward signature matches :class:`TraceKinV1` exactly so ``utils.trainer`` /
``training.trainer`` consume both interchangeably. Auxiliary loss tensors are
returned as zeros so the loss composition logic that adds them produces the
same numbers as a v1 run with the aux-loss weights set to 0.
"""
import math

import torch
import torch.nn.functional as F
import torch_geometric
from torch.nn import Embedding, Linear
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import softmax, to_dense_batch
from torch_scatter import scatter

from .drug_pool import MotifPool
from .layers import (
    AtomEncoder,
    Drug_PNAConv,
    DrugProteinConv,
    GCNCluster,
    MLP,
    PosLinear,
    Protein_PNAConv,
    dropout_edge,
)


EPS = 1e-15


class TraceKinV2(torch.nn.Module):
    """Redesigned TRACE-Kin model targeting the RF gap on catalytic kinetics."""

    def __init__(
        self,
        mol_deg,
        prot_deg,
        # Molecule
        mol_in_channels=43,
        prot_in_channels=33,
        prot_evo_channels=1280,
        hidden_channels=200,
        pre_layers=2,
        post_layers=1,
        aggregators=('mean', 'min', 'max', 'std'),
        scalers=('identity', 'amplification', 'linear'),
        # Interaction
        total_layer=3,
        K=(5, 10, 20),
        t=1,
        # Training
        heads=5,
        dropout=0,
        dropout_attn_score=0.2,
        drop_atom=0,
        drop_residue=0,
        dropout_cluster_edge=0,
        # Objective
        regression_head=True,
        classification_head=False,
        multiclassification_head=0,
        # v2-specific
        aa_residual_weight=0.1,
        shortcut_hidden=400,
        device='cuda:0',
    ):
        super().__init__()
        if not (regression_head or classification_head):
            raise ValueError("must have one objective")

        self.total_layer = total_layer
        self.regression_head = regression_head
        self.classification_head = classification_head
        self.multiclassification_head = multiclassification_head
        self.aa_residual_weight = float(aa_residual_weight)
        # learnable_aux_loss is exposed as False so trainer.py's
        # _compose_graph_reg_loss takes the simple summation branch.
        self.learnable_aux_loss = False

        if isinstance(K, int):
            K = [K] * total_layer
        K = list(K)

        # --- Atom / clique encoders (unchanged from v1) ---
        self.atom_type_encoder = Embedding(20, hidden_channels)
        self.atom_feat_encoder = MLP(
            [mol_in_channels, hidden_channels * 2, hidden_channels], out_norm=True
        )
        self.clique_encoder = Embedding(4, hidden_channels)

        # --- Protein input projections ---
        # Same shapes as v1 but used differently in forward (embedding-dominant).
        self.prot_evo = MLP(
            [prot_evo_channels, hidden_channels * 2, hidden_channels], out_norm=True
        )
        self.prot_aa = MLP(
            [prot_in_channels, hidden_channels * 2, hidden_channels], out_norm=True
        )

        # --- Per-layer GNN + cluster modules (mol/prot conv unchanged) ---
        self.mol_convs = torch.nn.ModuleList()
        self.prot_convs = torch.nn.ModuleList()
        self.mol_gn2 = torch.nn.ModuleList()
        self.prot_gn2 = torch.nn.ModuleList()
        self.inter_convs = torch.nn.ModuleList()
        self.cluster = torch.nn.ModuleList()
        self.mol_pools = torch.nn.ModuleList()
        self.mol_norms = torch.nn.ModuleList()
        self.prot_norms = torch.nn.ModuleList()
        self.atom_lins = torch.nn.ModuleList()
        self.residue_lins = torch.nn.ModuleList()
        self.c2a_mlps = torch.nn.ModuleList()
        self.c2r_mlps = torch.nn.ModuleList()

        self.num_cluster = K
        self.t = t
        self.prot_edge_dim = hidden_channels

        for idx in range(total_layer):
            self.mol_convs.append(Drug_PNAConv(
                mol_deg, hidden_channels, edge_channels=hidden_channels,
                pre_layers=pre_layers, post_layers=post_layers,
                aggregators=list(aggregators), scalers=list(scalers),
                num_towers=heads, dropout=dropout,
            ))
            self.prot_convs.append(Protein_PNAConv(
                prot_deg, hidden_channels, edge_channels=hidden_channels,
                pre_layers=pre_layers, post_layers=post_layers,
                aggregators=list(aggregators), scalers=list(scalers),
                num_towers=heads, dropout=dropout,
            ))
            self.cluster.append(GCNCluster(
                [hidden_channels, hidden_channels * 2, K[idx]], in_norm=True
            ))
            self.inter_convs.append(DrugProteinConv(
                atom_channels=hidden_channels,
                residue_channels=hidden_channels,
                heads=heads,
                t=t,
                dropout_attn_score=dropout_attn_score,
            ))
            self.mol_pools.append(MotifPool(hidden_channels, heads, dropout_attn_score, drop_atom))
            self.mol_norms.append(torch.nn.LayerNorm(hidden_channels))
            self.prot_norms.append(torch.nn.LayerNorm(hidden_channels))
            self.atom_lins.append(Linear(hidden_channels, hidden_channels, bias=False))
            self.residue_lins.append(Linear(hidden_channels, hidden_channels, bias=False))
            self.c2a_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))
            self.c2r_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))
            self.mol_gn2.append(GraphNorm(hidden_channels))
            self.prot_gn2.append(GraphNorm(hidden_channels))

        # --- Final attention-pooled output heads ---
        self.atom_attn_lin = PosLinear(heads * total_layer, 1, bias=False, init_value=1 / heads)
        self.residue_attn_lin = PosLinear(heads * total_layer, 1, bias=False, init_value=1 / heads)
        self.mol_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.prot_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)

        # --- v2 embedding shortcut branch ---
        # Mean-pooled raw embedding (B, prot_evo_channels) -> (B, hidden_channels).
        # The model concatenates this with the GNN protein output, so the regression
        # head input dim grows from 2*hidden to (2*hidden + hidden) == 3*hidden.
        self.shortcut_proj = MLP(
            [prot_evo_channels, shortcut_hidden, hidden_channels], out_norm=True
        )

        # Output heads sized for the concatenated representation.
        head_in = hidden_channels * 3  # mol(200) + gnn_prot(200) + shortcut(200)
        if self.regression_head:
            self.reg_out = MLP([head_in, hidden_channels, 1])
        if self.classification_head:
            self.cls_out = MLP([head_in, hidden_channels, 1])
        if self.multiclassification_head:
            self.mcls_out = MLP([head_in, hidden_channels, multiclassification_head])

        self.dropout = dropout
        self.drop_atom = drop_atom
        self.drop_residue = drop_residue
        self.dropout_cluster_edge = dropout_cluster_edge
        self.device = device

        # No learnable aux-loss parameters in v2 (aux losses are zero).
        self.loss_log_var_ortho = None
        self.loss_log_var_cluster = None
        self.loss_log_var_reg = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def reset_parameters(self):
        self.atom_feat_encoder.reset_parameters()
        self.prot_evo.reset_parameters()
        self.prot_aa.reset_parameters()
        self.shortcut_proj.reset_parameters()
        for idx in range(self.total_layer):
            self.mol_convs[idx].reset_parameters()
            self.prot_convs[idx].reset_parameters()
            self.mol_gn2[idx].reset_parameters()
            self.prot_gn2[idx].reset_parameters()
            self.cluster[idx].reset_parameters()
            self.mol_pools[idx].reset_parameters()
            self.mol_norms[idx].reset_parameters()
            self.prot_norms[idx].reset_parameters()
            self.inter_convs[idx].reset_parameters()
            self.atom_lins[idx].reset_parameters()
            self.residue_lins[idx].reset_parameters()
            self.c2a_mlps[idx].reset_parameters()
            self.c2r_mlps[idx].reset_parameters()
        self.atom_attn_lin.reset_parameters()
        self.residue_attn_lin.reset_parameters()
        self.mol_out.reset_parameters()
        self.prot_out.reset_parameters()
        if self.regression_head:
            self.reg_out.reset_parameters()
        if self.classification_head:
            self.cls_out.reset_parameters()
        if self.multiclassification_head:
            self.mcls_out.reset_parameters()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        # Molecule
        mol_x, mol_x_feat, bond_x, atom_edge_index,
        clique_x, clique_edge_index, atom2clique_index,
        # Protein
        residue_x, residue_evo_x, residue_edge_index, residue_edge_weight,
        # Mol-Protein interaction batch
        mol_batch=None, prot_batch=None, clique_batch=None,
        save_cluster=False,
    ):
        reg_pred = None
        cls_pred = None
        mcls_pred = None
        residue_edge_attr = _rbf(
            residue_edge_weight, D_max=1.0, D_count=self.prot_edge_dim, device=self.device
        )

        # --- v2 protein fusion: embedding-dominant ---
        h_evo = self.prot_evo(residue_evo_x)
        h_aa = self.prot_aa(residue_x)
        residue_x = h_evo + self.aa_residual_weight * h_aa

        # Molecule featurize (unchanged from v1)
        atom_x = self.atom_type_encoder(mol_x.squeeze()) + self.atom_feat_encoder(mol_x_feat)
        clique_x = self.clique_encoder(clique_x.squeeze())

        # Auxiliary losses are zero in v2 (returned for trainer compatibility).
        spectral_loss = torch.tensor(0.0, device=self.device)
        ortho_loss = torch.tensor(0.0, device=self.device)
        cluster_loss = torch.tensor(0.0, device=self.device)

        clique_scores = []
        residue_scores = []
        layer_s = {}

        for idx in range(self.total_layer):
            atom_x = self.mol_convs[idx](atom_x, bond_x, atom_edge_index)
            residue_x = self.prot_convs[idx](residue_x, residue_edge_index, residue_edge_attr)

            # Pool drug
            drug_x, clique_x, clique_score = self.mol_pools[idx](
                atom_x, clique_x, atom2clique_index, clique_batch, clique_edge_index
            )
            drug_x = self.mol_norms[idx](drug_x)
            clique_scores.append(clique_score)

            # --- v2 cluster aggregation (no MinCut loss) ---
            dropped_residue_edge_index, _ = dropout_edge(
                residue_edge_index, p=self.dropout_cluster_edge,
                force_undirected=True, training=self.training,
            )
            s_logits = self.cluster[idx](residue_x, dropped_residue_edge_index)  # (N, K)
            residue_hx, residue_mask = to_dense_batch(residue_x, prot_batch)     # (B, max_N, hidden)
            s_dense, _ = to_dense_batch(s_logits, prot_batch)                     # (B, max_N, K)

            if save_cluster:
                layer_s[idx] = s_dense

            # Soft assignment: softmax over K clusters per residue, then mask padding.
            s_soft = F.softmax(s_dense, dim=-1)
            s_soft = s_soft * residue_mask.unsqueeze(-1).float()

            # Cluster features are the residue-weighted sum (B, K, hidden).
            cluster_x = torch.bmm(s_soft.transpose(1, 2), residue_hx)
            cluster_x = self.prot_norms[idx](cluster_x)

            # Drug ↔ protein cross-attention (unchanged interface from v1)
            batch_size = s_soft.size(0)
            cluster_residue_batch = torch.arange(
                batch_size, device=self.device
            ).repeat_interleave(self.num_cluster[idx])
            cluster_x = cluster_x.reshape(batch_size * self.num_cluster[idx], -1)
            p2m_edge_index = torch.stack([
                torch.arange(batch_size * self.num_cluster[idx], device=self.device),
                torch.arange(batch_size, device=self.device).repeat_interleave(self.num_cluster[idx]),
            ])

            clique_x, cluster_x, inter_attn = self.inter_convs[idx](
                drug_x, clique_x, clique_batch, cluster_x, p2m_edge_index
            )
            inter_attn = inter_attn[1]

            # Residual: clique → atom
            row, col = atom2clique_index
            atom_x = atom_x + F.relu(self.atom_lins[idx](
                scatter(clique_x[col], row, dim=0, dim_size=atom_x.size(0), reduce='mean')
            ))
            atom_x = atom_x + self.c2a_mlps[idx](atom_x)
            atom_x = F.dropout(atom_x, self.dropout, training=self.training)

            # Residual: cluster → residue (use the same s_soft that did the pooling)
            residue_hx_back, _ = to_dense_batch(cluster_x, cluster_residue_batch)
            inter_attn_dense, _ = to_dense_batch(inter_attn, cluster_residue_batch)
            residue_x = residue_x + F.relu(self.residue_lins[idx](
                (s_soft @ residue_hx_back)[residue_mask]
            ))
            residue_x = residue_x + self.c2r_mlps[idx](residue_x)
            residue_x = F.dropout(residue_x, self.dropout, training=self.training)

            inter_attn = (s_soft @ inter_attn_dense)[residue_mask]
            residue_scores.append(inter_attn)

            atom_x = self.mol_gn2[idx](atom_x, mol_batch)
            residue_x = self.prot_gn2[idx](residue_x, prot_batch)

        # --- Final attention-weighted pooling (unchanged from v1) ---
        row, col = atom2clique_index
        clique_scores = torch.cat(clique_scores, dim=-1)
        atom_scores = scatter(
            clique_scores[col], row, dim=0, dim_size=atom_x.size(0), reduce='mean'
        )
        atom_score = self.atom_attn_lin(atom_scores)
        atom_score = softmax(atom_score, mol_batch)
        mol_pool_feat = global_add_pool(atom_x * atom_score, mol_batch)

        residue_scores = torch.cat(residue_scores, dim=-1)
        residue_score = softmax(self.residue_attn_lin(residue_scores), prot_batch)
        prot_pool_feat = global_add_pool(residue_x * residue_score, prot_batch)

        mol_pool_feat = self.mol_out(mol_pool_feat)
        prot_pool_feat = self.prot_out(prot_pool_feat)

        # --- v2 embedding shortcut branch ---
        # Mean-pool the raw 1280-dim embedding directly, project to hidden, concat.
        embedding_pool = scatter(residue_evo_x, prot_batch, dim=0, reduce='mean')
        embedding_shortcut = self.shortcut_proj(embedding_pool)

        # Concatenate: mol(200) + gnn_prot(200) + shortcut(200) = 600
        mol_prot_feat = torch.cat(
            [mol_pool_feat, prot_pool_feat, embedding_shortcut], dim=-1
        )

        if self.regression_head:
            reg_pred = self.reg_out(mol_prot_feat)
        if self.classification_head:
            cls_pred = self.cls_out(mol_prot_feat)
        if self.multiclassification_head:
            mcls_pred = self.mcls_out(mol_prot_feat)

        attention_dict = {
            'residue_final_score': residue_score,
            'atom_final_score': atom_score,
            'clique_layer_scores': clique_scores,
            'residue_layer_scores': residue_scores,
            'drug_atom_index': mol_batch,
            'drug_clique_index': clique_batch,
            'protein_residue_index': prot_batch,
            'mol_feature': mol_pool_feat,
            'prot_feature': prot_pool_feat,
            'embedding_shortcut_feature': embedding_shortcut,
            'interaction_fingerprint': mol_prot_feat,
            'cluster_s': layer_s,
        }

        return reg_pred, cls_pred, mcls_pred, spectral_loss, ortho_loss, cluster_loss, attention_dict

    # ------------------------------------------------------------------
    # Helpers (unchanged from v1)
    # ------------------------------------------------------------------
    def temperature_clamp(self):
        return

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, amsgrad):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, GraphNorm, PosLinear)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias') or pn.endswith('mean_scale'):
                    no_decay.add(fpn)
                elif 'loss_log_var' in pn:
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"params {inter_params} in both decay/no_decay"
        assert len(param_dict.keys() - union_params) == 0, (
            f"params {param_dict.keys() - union_params} not separated"
        )
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)

    def freeze_backbone_optimizers(
        self, finetune_module, weight_decay, learning_rate, betas, eps, amsgrad
    ):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, GraphNorm, PosLinear)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if not any(mn.startswith(name) for name in finetune_module):
                    p.requires_grad = False
                    continue
                p.requires_grad = True
                print(fpn, ' will be finetuned')
                if pn.endswith('bias') or pn.endswith('mean_scale'):
                    no_decay.add(fpn)
                elif 'loss_log_var' in pn:
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)


def _rbf(D, D_min=0.0, D_max=1.0, D_count=16, device='cpu'):
    """Radial basis function expansion of distances (from neurips19-graph-protein-design)."""
    D = torch.where(D < D_max, D, torch.tensor(D_max).float().to(device))
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
