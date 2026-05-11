import torch.utils.data
from torch_geometric.data import Dataset
# from torch.utils.data import Dataset
import torch
import pandas as pd
from torch_geometric.data import Data
import pickle
import torch.utils.data
from copy import deepcopy
import numpy as np


class ProteinMoleculeDataset(Dataset):
    def __init__(self, sequence_data, mol_obj, prot_obj, device='cpu', cache_transform=True):
        super(ProteinMoleculeDataset, self).__init__()

        if isinstance(sequence_data,pd.core.frame.DataFrame):
            self.pairs = sequence_data
        elif isinstance(sequence_data,str):
            self.pairs = pd.read_csv(sequence_data)
        else:
            raise Exception("provide dataframe object or csv path")
        
        ## MOLECULES
        if isinstance(mol_obj, dict):
            self.mols = mol_obj
        elif isinstance(mol_obj, str):
            with open(mol_obj, 'rb') as f:
                self.mols = pickle.load(f)
        else:
            raise Exception("provide dict mol object or pickle path")


        ## PROTEINS
        if isinstance(prot_obj, dict):
            self.prots = prot_obj
        elif isinstance(prot_obj, str):
            self.prots = torch.load(prot_obj)
        else:
            raise Exception("provide dict mol object or pickle path")

        self.device = device
        self.cache_transform = cache_transform

        if self.cache_transform:
            for _, v in self.mols.items():
                v['atom_idx'] = v['atom_idx'].long().view(-1, 1)
                v['atom_feature'] = v['atom_feature'].float()
                adj = v['bond_feature'].long()
                mol_edge_index =  adj.nonzero(as_tuple=False).t().contiguous()
                v['atom_edge_index'] = mol_edge_index
                v['atom_edge_attr'] = adj[mol_edge_index[0], mol_edge_index[1]].long()
                v['atom_num_nodes'] = v['atom_idx'].shape[0]

                ## Clique
                v['x_clique'] = v['x_clique'].long().view(-1, 1)
                v['clique_num_nodes'] = v['x_clique'].shape[0]
                v['tree_edge_index'] = v['tree_edge_index'].long()
                v['atom2clique_index'] = v['atom2clique_index'].long()

                ## v3 FP-MLP head: ChemBERT (768) embeddings live under
                ## `chembert_fp` and are populated by training/ligand_init.py
                ## from the per-row metabolite_features parquet column. No
                ## backfill — if the cache predates the ChemBERT cutover,
                ## rebuild via --force_rebuild. The v3 forward will raise
                ## loudly if chembert_fp is missing at __getitem__ time.

            for _, v in self.prots.items():
                v['seq_feat'] = v['seq_feat'].float()  # 33-d, ~50 KB per protein, OK to upcast
                # IMPORTANT: do NOT upcast token_representation here. At
                # MutaPLM's 4096-d × 6607 proteins × 400 res, that conversion
                # would balloon CPU RAM from ~21 GB (half) to ~43 GB (float).
                # Convert per-batch in __getitem__ instead — only ~100 MB per
                # batch with batch_size=16.
                v['num_nodes'] = len(v['seq'])
                v['node_pos'] = torch.arange(len(v['seq'])).reshape(-1,1)
                v['edge_weight'] = v['edge_weight'].float()

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return self.__len__()
    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        # Extract data
        mol_key = self.pairs.loc[idx,'Ligand']
        prot_key = self.pairs.loc[idx,'Protein'] 
        try: 
            reg_y = self.pairs.loc[idx,'regression_label'] 
            reg_y = torch.tensor(reg_y).float()
        except KeyError:
            reg_y = None
        

        try: 
            cls_y = self.pairs.loc[idx,'classification_label'] 
            cls_y = torch.tensor(cls_y).float()
        except KeyError:
            cls_y = None
        
        try: 
            mcls_y = self.pairs.loc[idx,'multiclass_label'] 
            mcls_y = torch.tensor(mcls_y + 1).float()
        except KeyError:
            mcls_y = None
        
        # Check if molecule and protein exist (defensive check)
        if mol_key not in self.mols:
            raise KeyError(f"Molecule '{mol_key}' not found in ligand dictionary. It may have been skipped due to invalid SMILES.")
        if prot_key not in self.prots:
            raise KeyError(f"Protein '{prot_key}' not found in protein dictionary.")
            
        mol = self.mols[mol_key]
        prot = self.prots[prot_key]
        
        ## PROT
        if self.cache_transform:
            ## atom
            mol_x = mol['atom_idx']
            mol_x_feat = mol['atom_feature']
            mol_edge_index  = mol['atom_edge_index']
            mol_edge_attr = mol['atom_edge_attr']
            mol_num_nodes = mol['atom_num_nodes']

            ## Clique
            mol_x_clique = mol['x_clique']
            clique_num_nodes = mol['clique_num_nodes']
            clique_edge_index = mol['tree_edge_index']
            atom2clique_index = mol['atom2clique_index']

            ## v3 FP-MLP ChemBERT embedding (per-graph, shape (1, 768))
            chembert_fp = mol['chembert_fp']
            ## Prot
            prot_seq = prot['seq']
            prot_node_aa = prot['seq_feat']
            # token_representation is stored at half precision in protein.pt
            # (~21 GB total for 6607 proteins at 4096-d MutaPLM). Convert
            # to float per-protein at __getitem__ time so the per-batch
            # GPU dtype matches the model's float32 weights, without
            # doubling CPU RAM by upcasting the whole cache at __init__.
            prot_node_evo = prot['token_representation'].float()
            prot_num_nodes = prot['num_nodes']
            prot_node_pos = prot['node_pos']
            prot_edge_index = prot['edge_index']
            prot_edge_weight = prot['edge_weight']
            ## v4 per-residue amino-acid index (L,) — small int tensor
            ## that the model uses to look up aa_typical means/stds from
            ## its own (n_aa+1, D) buffers. Only present when preprocess()
            ## was called with model_version="v4"; v1/v3 don't read it.
            prot_aa_idx = prot.get('prot_aa_idx')
        else:
            # MOL
            mol_x = mol['atom_idx'].long().view(-1, 1)
            mol_x_feat = mol['atom_feature'].float()
            adj = mol['bond_feature'].long()
            mol_edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            mol_edge_attr = adj[mol_edge_index[0], mol_edge_index[1]].long()
            mol_num_nodes = mol_x.shape[0]

            ## Clique
            mol_x_clique = mol['x_clique'].long().view(-1, 1)
            clique_num_nodes = mol_x_clique.shape[0]
            clique_edge_index = mol['tree_edge_index'].long()
            atom2clique_index = mol['atom2clique_index'].long()

            ## v3 FP-MLP ChemBERT embedding (per-graph, shape (1, 768))
            chembert_fp = mol['chembert_fp']

            prot_seq = prot['seq']
            prot_node_aa = prot['seq_feat'].float()
            prot_node_evo = prot['token_representation'].float()
            prot_num_nodes = len(prot['seq'])
            prot_node_pos = torch.arange(len(prot['seq'])).reshape(-1,1)
            prot_edge_index = prot['edge_index']
            prot_edge_weight = prot['edge_weight'].float()
            ## v4 per-residue amino-acid index
            prot_aa_idx = prot.get('prot_aa_idx')

        kwargs = dict(
                ## MOLECULE
                mol_x=mol_x, mol_x_feat=mol_x_feat, mol_edge_index=mol_edge_index,
                mol_edge_attr=mol_edge_attr, mol_num_nodes= mol_num_nodes,
                clique_x=mol_x_clique, clique_edge_index=clique_edge_index, atom2clique_index=atom2clique_index,
                clique_num_nodes=clique_num_nodes,
                ## v3 FP-MLP ChemBERT embedding (per-graph, shape (1, 768); batch to (B, 768))
                chembert_fp=chembert_fp,
                ## PROTEIN
                prot_node_aa=prot_node_aa, prot_node_evo=prot_node_evo,
                prot_node_pos=prot_node_pos, prot_seq=prot_seq,
                prot_edge_index=prot_edge_index, prot_edge_weight=prot_edge_weight,
                prot_num_nodes=prot_num_nodes,
                ## Y output
                reg_y=reg_y, cls_y=cls_y, mcls_y=mcls_y,
                ## keys
                mol_key = mol_key, prot_key = prot_key,
        )
        ## v4 per-residue amino-acid index (L,) — int tensor, batched
        ## along dim=0 by PyG to (sum_L,). Only attached when preprocess()
        ## was called with model_version="v4". The model uses these to
        ## index_select per-residue means/stds from its own (n_aa+1, D)
        ## buffers, avoiding the per-protein (L, D) duplication that
        ## OOM'd at D=4096.
        if prot_aa_idx is not None:
            kwargs['prot_aa_idx'] = prot_aa_idx
        return MultiGraphData(**kwargs)

def maybe_num_nodes(index, num_nodes=None):
    # NOTE(WMF): I find out a problem here, 
    # index.max().item() -> int
    # num_nodes -> tensor
    # need type conversion.
    # return index.max().item() + 1 if num_nodes is None else num_nodes
    return index.max().item() + 1 if num_nodes is None else int(num_nodes)

def get_self_loop_attr(edge_index, edge_attr, num_nodes):
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    .. note::
        This operation is analogous to getting the diagonal elements of the
        dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = torch.tensor([0.2, 0.3, 0.5])
        >>> get_self_loop_attr(edge_index, edge_weight)
        tensor([0.5000, 0.0000])

        >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
        tensor([0.5000, 0.0000, 0.0000, 0.0000])
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = torch.ones_like(loop_index, dtype=torch.float)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = loop_attr.new_zeros((num_nodes, ) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr



class MultiGraphData(Data):
    def __inc__(self, key, item, *args):
        if key == 'mol_edge_index':
            return self.mol_x.size(0)
        elif key == 'clique_edge_index':
            return self.clique_x.size(0)
        elif key == 'atom2clique_index':
            return torch.tensor([[self.mol_x.size(0)], [self.clique_x.size(0)]])
        elif key == 'prot_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'prot_struc_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'm2p_edge_index':
             return torch.tensor([[self.mol_x.size(0)], [self.prot_node_aa.size(0)]])
        # elif key == 'edge_index_p2m':
        #     return torch.tensor([[self.prot_node_s.size(0)],[self.mol_x.size(0)]])
        else:
            return super(MultiGraphData, self).__inc__(key, item, *args)

