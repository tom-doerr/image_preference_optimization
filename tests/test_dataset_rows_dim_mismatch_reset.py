import os
import numpy as np


def test_append_resets_on_dim_mismatch(tmp_path, monkeypatch):
    # Use a unique prompt so we don't touch existing files
    prompt = "dim-mismatch-reset-test"
    # Point CWD to temp to isolate dataset files
    monkeypatch.chdir(tmp_path)
    from persistence import dataset_path_for_prompt, dataset_rows_for_prompt, append_dataset_row

    path = dataset_path_for_prompt(prompt)
    # Seed an existing dataset with d=16384 (e.g., 512x512)
    X_old = np.zeros((2, 16384), dtype=float)
    y_old = np.array([1.0, -1.0], dtype=float)
    np.savez_compressed(path, X=X_old, y=y_old)
    assert dataset_rows_for_prompt(prompt) == 2

    # Now append a row with a different dim (e.g., 448x448 → 12544)
    feat_new = np.zeros((1, 12544), dtype=float)
    n = append_dataset_row(prompt, feat_new, +1.0)
    # After mismatch, we start a fresh aggregate file for the new dim
    assert n == 1
    assert dataset_rows_for_prompt(prompt) == 1

