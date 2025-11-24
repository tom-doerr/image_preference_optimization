from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Optional

import numpy as np
from ipo.core.persistence import get_dataset_for_prompt_or_session


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]


def train_xgb_for_prompt(
    prompt: str,
    n_estimators: int = 50,
    max_depth: int = 3,
    save_model: bool = True,
    model_dir: str = ".",
) -> Optional[str]:
    """Train an XGBoost classifier on the folder-backed dataset for the prompt.

    Returns the saved model path when save_model is True, otherwise None.
    """
    X, y = get_dataset_for_prompt_or_session(prompt, type("SS", (), {})())
    if X is None or y is None or X.shape[0] == 0:
        raise ValueError("Dataset is empty; add samples first.")
    if len(set(np.asarray(y).astype(int))) < 2:
        raise ValueError("Need both classes to train XGBoost (labels +1 and -1).")

    from ipo.core.xgb_value import fit_xgb_classifier  # local import keeps CLI light

    model = fit_xgb_classifier(X, y, n_estimators=n_estimators, max_depth=max_depth)
    preds = model.predict_proba(X)[:, 1] >= 0.5
    acc = float(np.mean(preds == (np.asarray(y) > 0)))

    out_path = None
    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, f"xgb_model_{_prompt_hash(prompt)}.bin")
        model.save_model(out_path)
        print(f"[xgb-cli] saved model to {out_path}")
    print(f"[xgb-cli] trained rows={X.shape[0]} d={X.shape[1]} acc={acc:.3f}")
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train XGBoost on the current prompt dataset."
    )
    ap.add_argument(
        "--prompt", required=True, help="Prompt string used for dataset hashing"
    )
    ap.add_argument("--n-estimators", type=int, default=50, help="Number of trees")
    ap.add_argument("--max-depth", type=int, default=3, help="Tree depth")
    ap.add_argument("--no-save", action="store_true", help="Do not write a model file")
    ap.add_argument(
        "--model-dir", default=".", help="Output directory for the model file"
    )
    args = ap.parse_args(argv)
    train_xgb_for_prompt(
        args.prompt,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        save_model=not args.no_save,
        model_dir=args.model_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
try:
    from rich_cli import enable_color_print as _enable_color

    _enable_color()
except Exception:
    pass
