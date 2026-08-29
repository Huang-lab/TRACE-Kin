"""Static checks that the repo's entry points agree with what the model package exports.

These parse the source with `ast` instead of importing it, so they need no torch,
torch_geometric or rdkit and run in a second. That matters here: importing
`inference/trace_kin_predictor.py` needs the full GPU stack, which is why nobody
noticed that its top-level import has named a class that does not exist since
`ecef1b5` ("Replace v2 with v3 dual-head + learned gate architecture").

`dfb7ab3a` ("got rid of unused older versions") then deleted five more model
classes, and `training/v6b_xgboost_on_v5t_features.py` still imports one of them.
Both files fail at import, not at some argument-dependent branch, so there is no
configuration under which they run.
"""
import ast
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
MODEL_PKG = "models.trace_kin"


def _exported_names() -> set:
    """`__all__` from models/trace_kin/__init__.py, read without importing it."""
    tree = ast.parse((REPO / "models" / "trace_kin" / "__init__.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return {elt.value for elt in node.value.elts}
    raise AssertionError("models/trace_kin/__init__.py defines no __all__")


# Files whose top-level import already names a deleted class. Both need a
# decision rather than a mechanical fix -- see the linked issue -- so they are
# recorded here instead of silently excluded. strict=True means that repairing
# one turns the test red as XPASS, which is the prompt to delete its entry.
KNOWN_BROKEN = {
    "inference/trace_kin_predictor.py":
        "imports TraceKinV1/TraceKinV2; V2 went in ecef1b5, V1 in dfb7ab3a. "
        "The inference entry point cannot be imported at all. Rewriting it "
        "against V7 is an architecture decision.",
    "training/v6b_xgboost_on_v5t_features.py":
        "imports TraceKinV5T, deleted in dfb7ab3a. The v5t feature extractor "
        "this script wraps no longer exists.",
}


def _python_files() -> list:
    skip = {".git", "__pycache__", "tests"}
    params = []
    for path in sorted(REPO.rglob("*.py")):
        if any(part in skip for part in path.parts):
            continue
        rel = str(path.relative_to(REPO))
        marks = ([pytest.mark.xfail(strict=True, reason=KNOWN_BROKEN[rel])]
                 if rel in KNOWN_BROKEN else [])
        params.append(pytest.param(path, id=rel, marks=marks))
    return params


@pytest.mark.parametrize("path", _python_files())
def test_model_imports_name_something_that_exists(path):
    """Every `from models.trace_kin import X` must name an exported class.

    A miss here is not a latent edge case: the import is at module scope, so the
    file cannot be run at all.
    """
    exported = _exported_names()
    tree = ast.parse(path.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == MODEL_PKG:
            imported = {alias.name for alias in node.names}
            missing = sorted(imported - exported)
            assert not missing, (
                f"{path.relative_to(REPO)}:{node.lineno} imports {missing} from "
                f"{MODEL_PKG}, which exports only {sorted(exported)}. "
                "The file cannot be imported."
            )


def test_cli_offers_only_buildable_model_versions():
    """`--model_version` must not advertise architectures build_model cannot build."""
    src = (REPO / "training" / "train_trace_kin.py").read_text()
    tree = ast.parse(src)
    ns = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                        "AVAILABLE_MODEL_VERSIONS", "REMOVED_MODEL_VERSIONS"):
                    ns[target.id] = ast.literal_eval(node.value)

    assert "AVAILABLE_MODEL_VERSIONS" in ns, "train_trace_kin.py must declare the buildable set"
    assert not (ns["AVAILABLE_MODEL_VERSIONS"] & ns["REMOVED_MODEL_VERSIONS"]), \
        "a version cannot be both available and removed"

    # Every available version needs a branch in build_model.
    for version in ns["AVAILABLE_MODEL_VERSIONS"]:
        assert f'version == "{version}"' in src, \
            f"{version} is offered but build_model has no branch for it"


def test_default_model_version_is_buildable():
    """`model_config.get("model_version", ...)` used to default to "v1", which was
    deleted -- so a config without the key selected a branch that could only fail."""
    src = (REPO / "training" / "train_trace_kin.py").read_text()
    tree = ast.parse(src)

    defaults = [
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "model_version"
        and isinstance(node.args[1], ast.Constant)
    ]
    assert defaults, "no defaulted model_version lookup found"

    ns = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "AVAILABLE_MODEL_VERSIONS":
                    ns = ast.literal_eval(node.value)
    for default in defaults:
        assert default in ns, f"model_version defaults to {default!r}, which cannot be built"
