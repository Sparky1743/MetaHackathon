from pathlib import Path

import nbformat


def test_training_notebooks_are_parseable_and_point_to_scripts():
    notebooks = {
        "01_smoke_test.ipynb": "OnCallRedShiftEnv",
        "02_train_grpo_unsloth.ipynb": "train_redshift_policy.py",
        "03_eval_baseline_vs_trained.ipynb": "evaluate_redshift_policy.py",
    }
    for name, expected in notebooks.items():
        nb = nbformat.read(Path("notebooks") / name, as_version=4)
        source = "\n".join(cell.source for cell in nb.cells)
        assert expected in source

