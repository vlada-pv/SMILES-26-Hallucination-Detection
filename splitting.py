"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    The default strategy performs a single stratified random split preserving
    the class ratio in each subset.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    idx = np.arange(len(y))

    # Prefer stratified K-fold for more stable evaluation.
    # Additionally, if prompts repeat, we keep identical prompts in the same fold
    # to reduce leakage (group-aware splitting).
    n_splits = int(os.getenv("SPLIT_N_SPLITS", "5"))
    n_splits = max(2, n_splits)
    groups = None
    if df is not None and "prompt" in df.columns:
        prompts = df["prompt"].astype(str).values
        # Use prompt string as group id (factorized).
        groups = pd.factorize(prompts)[0]

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    if groups is not None:
        # StratifiedGroupKFold is ideal if available; fall back to StratifiedKFold otherwise.
        try:
            from sklearn.model_selection import StratifiedGroupKFold

            sgkf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            outer_iter = sgkf.split(idx, y, groups=groups)
        except Exception:
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            outer_iter = skf.split(idx, y)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        outer_iter = skf.split(idx, y)

    for idx_train_val, idx_test in outer_iter:
        # Keep a validation split inside each fold for threshold tuning.
        relative_val = val_size / (1.0 - test_size)
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=relative_val,
            random_state=random_state,
            stratify=y[idx_train_val],
        )
        splits.append((idx_train, idx_val, idx_test))

    return splits

