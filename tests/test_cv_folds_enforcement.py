"""T1.7 — Static check that training scripts never create their own CV splits.

All fold assignments must come from data/cv_folds.json via load_cv_folds().
Creating new splits mid-run would make model comparisons invalid because
different models would be evaluated on different data.
"""

import ast
from pathlib import Path


TRAINING_SCRIPTS = [
    "scripts/train_baselines.py",
    "scripts/train_fusion.py",
    "scripts/train_pathway_fusion.py",
    "scripts/run_ablations.py",
]

# Any call to these sklearn splitter constructors in training scripts is a bug
FORBIDDEN_SPLITTERS = {
    "KFold",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
    "ShuffleSplit",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "train_test_split",
    "GroupKFold",
    "GroupShuffleSplit",
}


def _get_all_call_names(tree: ast.AST) -> set[str]:
    """Return set of function/class call names found in the AST."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def test_training_scripts_do_not_create_own_cv_splits():
    """No training script may construct its own fold splits."""
    violations = []
    for script_path_str in TRAINING_SCRIPTS:
        script_path = Path(script_path_str)
        if not script_path.exists():
            continue
        source = script_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=script_path_str)
        except SyntaxError:
            continue
        calls = _get_all_call_names(tree)
        found = calls & FORBIDDEN_SPLITTERS
        if found:
            violations.append(f"{script_path_str}: {sorted(found)}")

    assert not violations, (
        "Training scripts must use load_cv_folds() from data/cv_folds.json. "
        "Found forbidden splitter calls:\n" + "\n".join(violations)
    )


def test_training_scripts_all_import_load_cv_folds():
    """Training scripts that build fold loops must call load_cv_folds."""
    missing = []
    for script_path_str in TRAINING_SCRIPTS:
        script_path = Path(script_path_str)
        if not script_path.exists():
            continue
        source = script_path.read_text(encoding="utf-8")
        # Only check scripts that actually iterate folds
        if "fold" not in source.lower():
            continue
        if "load_cv_folds" not in source:
            missing.append(script_path_str)

    assert not missing, (
        "These training scripts iterate folds but do not call load_cv_folds():\n"
        + "\n".join(missing)
    )
