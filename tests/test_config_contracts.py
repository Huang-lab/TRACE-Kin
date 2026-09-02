"""Static checks that a model option can be reached from a config file.

Three times now an option has been added to the model and left unreachable, and
each time the code looked correct at every individual site:

* `dfb7ab3a` deleted `"mol_pe_mode"` from `config_v7.json`. `net_v7.py` defaults
  it to `"none"`, so the ligand PE the milestone was about was silently off
  (issue #5).
* `4f5c231` added `pe_raw_norm` to `LigandEncoder.__init__` and the BatchNorm it
  gates in `forward`, but nothing passes it. `net_v7.py` calls `LigandEncoder`
  with seven keyword arguments and that is not one of them, so the parameter
  keeps its `"none"` default on every path and the commit is a no-op in every
  configuration a training run can produce.
* `mol_pe_fold_norm` reached the constructor but never the config, so which of
  two disagreeing reference implementations the run followed was recorded only
  as a Python default.

These parse the source with `ast` rather than importing it, so they need no
torch, torch_geometric or rdkit and run in well under a second -- which is the
only reason a check like this gets run at all. Importing this package needs the
full GPU stack, which is exactly why an unreachable keyword argument is
invisible: there is no cheap way to look.

The rule they encode: **an option that changes the model has to be reachable
from the file that records the run.** Config key -> `build_model` -> model
`__init__` -> encoder `__init__`, with no gap.
"""
import ast
import json
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
NET_V7 = REPO / "models" / "trace_kin" / "net_v7.py"
LIGAND_ENCODER = REPO / "models" / "trace_kin" / "ligand_encoder.py"
TRAINER = REPO / "training" / "train_trace_kin.py"
CONFIG_V7 = REPO / "training" / "config_v7.json"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _init_params(path: Path, class_name: str) -> list:
    """Parameter names of `class_name.__init__`, `self` excluded."""
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = item.args
                    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
                    return [n for n in names if n != "self"]
    raise AssertionError(f"{class_name}.__init__ not found in {path.name}")


def _call_keywords(path: Path, callee: str) -> set:
    """Keyword argument names at every `callee(...)` call site in `path`."""
    found = set()
    seen_call = False
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != callee:
            continue
        seen_call = True
        found |= {kw.arg for kw in node.keywords if kw.arg is not None}
    if not seen_call:
        raise AssertionError(f"no call to {callee}() in {path.name}")
    return found


def _params_get_keys(path: Path) -> set:
    """Every `params.get("key", ...)` key read in `path`."""
    keys = set()
    for node in ast.walk(_tree(path)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "params"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
    return keys


def _config_params() -> dict:
    return json.loads(CONFIG_V7.read_text())["params"]


# The ligand positional-encoding options. Named explicitly rather than derived,
# so that adding an option to the encoder without adding it here is itself the
# thing that has to be a deliberate act.
PE_OPTIONS = ["pe_mode", "pe_steps", "pe_dim", "pe_fold_norm", "pe_raw_norm"]


@pytest.mark.parametrize("option", PE_OPTIONS)
def test_encoder_option_is_passed_by_net_v7(option):
    """Every PE option `LigandEncoder` accepts is passed where it is built.

    A parameter with a default that no caller overrides is not an option; it is
    a constant with a misleading name. `pe_raw_norm` was one from `4f5c231`
    until this test.
    """
    assert option in _init_params(LIGAND_ENCODER, "LigandEncoder"), (
        f"{option} is not a LigandEncoder parameter; update PE_OPTIONS"
    )
    passed = _call_keywords(NET_V7, "LigandEncoder")
    assert option in passed, (
        f"LigandEncoder({option}=...) is never passed in {NET_V7.name}, so the "
        f"parameter keeps its default in every configuration. Passed: {sorted(passed)}"
    )


@pytest.mark.parametrize("option", PE_OPTIONS)
def test_net_v7_option_is_read_from_the_config(option):
    """Every `mol_pe_*` parameter of the model is read from `params`.

    The trainer is the only thing that turns a config file into a model, so a
    model parameter it never reads cannot be set by a config, whatever the
    config says.
    """
    key = f"mol_{option}"
    assert key in _init_params(NET_V7, "TraceKinV7"), (
        f"{key} is not a TraceKinV7 parameter"
    )
    read = _params_get_keys(TRAINER)
    assert key in read, (
        f"build_model never reads params[{key!r}], so config_v7.json cannot set it. "
        f"mol_pe_* keys it does read: {sorted(k for k in read if k.startswith('mol_pe'))}"
    )


@pytest.mark.parametrize("option", PE_OPTIONS)
def test_config_v7_records_the_option_explicitly(option):
    """`config_v7.json` states each PE option rather than inheriting a default.

    This is the one that would have caught issue #5. A run's config file is the
    only record of what that run was; an option that is live in the experiment
    and absent from the file is one whose value nobody can recover afterwards,
    and `dfb7ab3a` showed that "absent" and "off" are indistinguishable in the
    log.

    `mol_pe_fold_norm` is the sharper case: the two gnn-lspe reference files
    disagree about whether to normalise, `a8cbf5e` exposed the choice as a
    parameter precisely because they disagree, and then left the answer in a
    Python default.
    """
    key = f"mol_{option}"
    params = _config_params()
    assert key in params, (
        f"config_v7.json does not set {key!r}, so its value comes from a default "
        f"in build_model and nothing in the repo records what the run used"
    )


def test_the_config_sets_no_pe_key_the_trainer_ignores():
    """The reverse direction: a key in the file that nothing reads.

    Harmless to the run and actively misleading to a reader, who has no way to
    tell a live setting from a typo. `"mol_pe_step": 8` would train at
    `mol_pe_steps=8`'s default and look deliberate.
    """
    read = _params_get_keys(TRAINER)
    orphans = sorted(k for k in _config_params() if k.startswith("mol_pe") and k not in read)
    assert not orphans, f"config_v7.json sets keys build_model never reads: {orphans}"


def test_lspe_only_layers_are_built_only_for_lspe():
    """`p_out` / `Whp` are constructed under a `pe_mode == "lspe"` guard.

    They are used only where `p` exists. Built unconditionally they take no
    gradient in the other two modes -- so they sit at initialisation -- and they
    enter the `state_dict` of a `pe_mode="none"` model, which means a baseline
    checkpoint saved before `a8cbf5e` no longer loads into a baseline model.
    The commit's own note says it "affects mol_pe_mode='lspe' only"; this is the
    check that makes that true.
    """
    src = _tree(LIGAND_ENCODER)
    for node in ast.walk(src):
        if not (isinstance(node, ast.ClassDef) and node.name == "LigandEncoder"):
            continue
        for item in node.body:
            if not (isinstance(item, ast.FunctionDef) and item.name == "__init__"):
                continue
            for stmt in item.body:
                assigned = {
                    t.attr
                    for n in ast.walk(stmt)
                    if isinstance(n, ast.Assign)
                    for t in n.targets
                    if isinstance(t, ast.Attribute)
                }
                if not {"p_out", "Whp"} & assigned:
                    continue
                assert isinstance(stmt, ast.If), (
                    "self.p_out / self.Whp are assigned outside any `if`, so "
                    "they exist for every pe_mode"
                )
                guard = ast.unparse(stmt.test)
                assert "lspe" in guard, (
                    f"p_out/Whp are guarded by {guard!r}, which does not mention lspe"
                )
                return
    raise AssertionError("no assignment to self.p_out found in LigandEncoder.__init__")
