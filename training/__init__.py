"""TRACE-Kin training package.

A slim port of the original PSICHIC training pipeline that produced the
benchmark in ``trace_doc/kinetic_regress_benchmark.csv``. Two architecture
variants are supported via ``--model_version v1|v2``:

- v1: original PSICHIC, frozen baseline for A/B comparison.
- v2: redesigned architecture targeting the RF gap on catalytic kinetics.

Cross-dataset training pooling for catalytic kinetics is supported at the
DataFrame layer in ``train_trace_kin.py`` via ``--pool_train_csvs``. SWA
(Stochastic Weight Averaging) is supported in :class:`Trainer` via the
``use_swa`` constructor argument.
"""

from .data_utils import (
    DataLoader,
    CustomWeightedRandomSampler,
    compute_pna_degrees,
    sampler_from_weights,
    unbatch,
    virtual_screening,
)
from .dataset import ProteinMoleculeDataset, MultiGraphData
from .ligand_init import ligand_init, smiles2graph
from .metrics import evaluate_cls, evaluate_mcls, evaluate_reg
from .protein_init_with_embedding import protein_init_with_embedding
from .trainer import Trainer

__all__ = [
    "DataLoader",
    "CustomWeightedRandomSampler",
    "compute_pna_degrees",
    "sampler_from_weights",
    "unbatch",
    "virtual_screening",
    "ProteinMoleculeDataset",
    "MultiGraphData",
    "ligand_init",
    "smiles2graph",
    "evaluate_cls",
    "evaluate_mcls",
    "evaluate_reg",
    "protein_init_with_embedding",
    "Trainer",
]
