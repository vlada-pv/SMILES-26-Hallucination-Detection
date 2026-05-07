"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``; the default architecture is a single
    hidden-layer MLP with ``StandardScaler`` pre-processing.  The network is
    built lazily in ``fit()`` once the feature dimension is known.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None  # built lazily in fit()
        self._nets: list[nn.Sequential] = []
        self._scaler = StandardScaler()
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()
        self._xgb = None
        self._model_kind: str = "mlp"
        self._tree_models: list[object] = []
        self._tree_stacker: LogisticRegression | None = None

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the network definition below.
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Called once at the start of ``fit()`` when ``input_dim`` is known.

        Args:
            input_dim: Feature vector dimensionality.
        """
        hidden_dim = int(os.getenv("PROBE_HIDDEN_DIM", "192"))
        p_drop = float(os.getenv("PROBE_DROPOUT", "0.25"))
        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw (pre-sigmoid) logits.
        """
        if self._model_kind != "mlp":
            raise RuntimeError("forward() is only supported for the MLP probe.")
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``, builds the network if needed,
        and optimises with Adam + ``BCEWithLogitsLoss``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        self._model_kind = os.getenv("PROBE_MODEL", "mlp").lower().strip()

        # Always scale (helps both MLP and linear baselines; harmless for XGB).
        X_scaled = self._scaler.fit_transform(X)

        if self._model_kind in {"xgb", "xgboost"}:
            try:
                from xgboost import XGBClassifier
            except Exception as e:  # pragma: no cover
                print(f"[XGB] unavailable, falling back to ExtraTrees. Reason: {e}")
                self._model_kind = "et"

            n_pos = int(y.sum())
            n_neg = int(len(y) - n_pos)
            scale_pos_weight = float(n_neg / max(n_pos, 1))

            # Conservative defaults aimed at reducing overfitting.
            max_depth = int(os.getenv("XGB_MAX_DEPTH", "5"))
            n_estimators = int(os.getenv("XGB_N_ESTIMATORS", "600"))
            learning_rate = float(os.getenv("XGB_LR", "0.05"))
            subsample = float(os.getenv("XGB_SUBSAMPLE", "0.8"))
            colsample = float(os.getenv("XGB_COLSAMPLE", "0.8"))
            reg_alpha = float(os.getenv("XGB_REG_ALPHA", "0.0"))
            reg_lambda = float(os.getenv("XGB_REG_LAMBDA", "1.0"))
            min_child_weight = float(os.getenv("XGB_MIN_CHILD_WEIGHT", "5.0"))

            X_tr, X_es, y_tr, y_es = train_test_split(
                X_scaled, y, test_size=0.15, random_state=42, stratify=y
            )

            if self._model_kind in {"xgb", "xgboost"}:
                try:
                    self._xgb = XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        min_child_weight=min_child_weight,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=int(os.getenv("PROBE_SEED", "42")),
                        scale_pos_weight=scale_pos_weight,
                    )

                    self._xgb.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_es, y_es)],
                        verbose=False,
                    )

                    # Light-weight importance peek (first fit only).
                    try:
                        importances = np.asarray(
                            self._xgb.feature_importances_, dtype=float
                        )
                        if importances.size:
                            topk = min(10, importances.size)
                            top_idx = np.argsort(importances)[::-1][:topk]
                            top = [(int(i), float(importances[i])) for i in top_idx]
                            print(f"[XGB] top feature importances (idx, gain): {top}")
                    except Exception:
                        pass

                    # Keep torch-related members in a consistent "not used" state.
                    self._nets = []
                    self._net = None
                    return self
                except Exception as e:  # pragma: no cover
                    # Common on macOS when libomp is missing.
                    print(f"[XGB] failed to run, falling back to ExtraTrees. Reason: {e}")
                    self._model_kind = "et"

        if self._model_kind in {"et", "extratrees", "trees"}:
            from sklearn.ensemble import ExtraTreesClassifier

            n_estimators = int(os.getenv("ET_N_ESTIMATORS", "600"))
            max_depth_raw = os.getenv("ET_MAX_DEPTH", "10").strip().lower()
            max_depth = None if max_depth_raw in {"none", ""} else int(max_depth_raw)
            min_samples_leaf = int(os.getenv("ET_MIN_SAMPLES_LEAF", "5"))
            min_samples_split = int(os.getenv("ET_MIN_SAMPLES_SPLIT", "2"))
            max_features = os.getenv("ET_MAX_FEATURES", "sqrt")
            bootstrap = os.getenv("ET_BOOTSTRAP", "0").strip().lower() in {"1", "true", "yes"}

            # Ensemble across seeds improves stability and usually bumps AUROC a bit.
            seeds_raw = os.getenv("PROBE_ENSEMBLE_SEEDS", "41,42,43")
            seeds = [int(s.strip()) for s in seeds_raw.split(",") if s.strip()]
            if not seeds:
                seeds = [int(os.getenv("PROBE_SEED", "42"))]

            use_stacking = os.getenv("PROBE_TREE_STACKING", "1").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            self._tree_stacker = None

            # Internal split for optional light stacking/calibration.
            idx = np.arange(len(y))
            idx_tr, idx_es = train_test_split(
                idx, test_size=0.15, random_state=42, stratify=y
            )
            X_tr, y_tr = X_scaled[idx_tr], y[idx_tr]
            X_es, y_es = X_scaled[idx_es], y[idx_es]

            self._tree_models = []
            for seed in seeds:
                model = ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split,
                    max_features=max_features,
                    bootstrap=bootstrap,
                    n_jobs=-1,
                    random_state=seed,
                    class_weight="balanced_subsample",
                )
                # Fit on internal train for stacking signal.
                model.fit(X_tr, y_tr)
                self._tree_models.append(model)

            if use_stacking:
                # Stack ET averaged probability with a tiny logistic layer.
                es_probs = [m.predict_proba(X_es)[:, 1] for m in self._tree_models]
                es_avg = np.mean(np.stack(es_probs, axis=0), axis=0)
                self._tree_stacker = LogisticRegression(
                    C=float(os.getenv("TREE_STACKER_C", "1.0")),
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=42,
                    max_iter=2000,
                )
                self._tree_stacker.fit(es_avg.reshape(-1, 1), y_es)

            # Refit ET ensemble on full fold train for stronger final model.
            self._tree_models = []
            for seed in seeds:
                model = ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split,
                    max_features=max_features,
                    bootstrap=bootstrap,
                    n_jobs=-1,
                    random_state=seed,
                    class_weight="balanced_subsample",
                )
                model.fit(X_scaled, y)
                self._tree_models.append(model)

            # For backward-compat: keep _xgb pointing to the first model.
            self._xgb = self._tree_models[0] if self._tree_models else None

            try:
                importances = np.asarray(self._xgb.feature_importances_, dtype=float)
                if importances.size:
                    topk = min(10, importances.size)
                    top_idx = np.argsort(importances)[::-1][:topk]
                    top = [(int(i), float(importances[i])) for i in top_idx]
                    print(f"[ET] top feature importances (idx, impurity): {top}")
            except Exception:
                pass

            self._nets = []
            self._net = None
            return self

        # -------------------- MLP (existing default) --------------------
        X_t_full = torch.from_numpy(X_scaled).float()
        y_t_full = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Small ensemble across seeds to stabilize predictions.
        seeds_raw = os.getenv("PROBE_ENSEMBLE_SEEDS", "41,42,43")
        seeds = [int(s.strip()) for s in seeds_raw.split(",") if s.strip()]
        self._nets = []

        # Internal early-stopping split.
        idx = np.arange(len(y))
        idx_tr, idx_es = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)
        X_tr = X_t_full[idx_tr]
        y_tr = y_t_full[idx_tr]
        X_es = X_t_full[idx_es]
        y_es = y_t_full[idx_es]

        learning_rate = float(os.getenv("PROBE_LR", "8e-4"))
        n_epochs = int(os.getenv("PROBE_EPOCHS", "260"))
        batch_size = int(os.getenv("PROBE_BATCH_SIZE", "64"))
        patience = int(os.getenv("PROBE_PATIENCE", "20"))
        weight_decay = float(os.getenv("PROBE_WEIGHT_DECAY", "1e-4"))

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            self._build_network(X_scaled.shape[1])
            assert self._net is not None
            net = self._net
            optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

            best_state = None
            best_loss = float("inf")
            bad = 0

            for _ in range(n_epochs):
                net.train()
                perm = torch.randperm(X_tr.size(0))
                for start in range(0, X_tr.size(0), batch_size):
                    b = perm[start : start + batch_size]
                    xb = X_tr[b]
                    yb = y_tr[b]

                    optimizer.zero_grad()
                    loss = criterion(net(xb).squeeze(-1), yb)
                    loss.backward()
                    optimizer.step()

                net.eval()
                with torch.no_grad():
                    es_loss = float(criterion(net(X_es).squeeze(-1), y_es).item())

                if es_loss + 1e-6 < best_loss:
                    best_loss = es_loss
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self._nets.append(net)

        # Keep forward() compatible.
        self._net = self._nets[0] if self._nets else self._net
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        # Candidate thresholds:
        # - unique predicted probabilities
        # - midpoints between consecutive unique probabilities (important for accuracy)
        # - a coarse grid for robustness
        uniq = np.unique(probs)
        if uniq.size >= 2:
            mids = (uniq[:-1] + uniq[1:]) / 2.0
            candidates = np.unique(
                np.concatenate([uniq, mids, np.linspace(0.0, 1.0, 101)])
            )
        else:
            candidates = np.unique(np.concatenate([uniq, np.linspace(0.0, 1.0, 101)]))

        metric = os.getenv("PROBE_THRESHOLD_METRIC", "accuracy").lower()
        best_threshold = 0.5
        best_f1 = -1.0
        best_acc = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            acc = (y_pred_t == y_val).mean()
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if metric == "accuracy":
                if acc > best_acc or (acc == best_acc and score > best_f1):
                    best_acc = float(acc)
                    best_f1 = score
                    best_threshold = float(t)
            elif score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_scaled = self._scaler.transform(X)

        if self._model_kind in {"xgb", "xgboost"}:
            if self._xgb is None:
                raise RuntimeError("Classifier is not fitted yet. Call fit() first.")
            # XGB returns [P(y=0), P(y=1)] for binary.
            return self._xgb.predict_proba(X_scaled)
        if self._model_kind in {"et", "extratrees", "trees"}:
            if self._xgb is None:
                raise RuntimeError("Classifier is not fitted yet. Call fit() first.")
            if self._tree_models:
                probs = [m.predict_proba(X_scaled)[:, 1] for m in self._tree_models]
                avg_pos = np.mean(np.stack(probs, axis=0), axis=0)
                if self._tree_stacker is not None:
                    stacked = self._tree_stacker.predict_proba(avg_pos.reshape(-1, 1))
                    return stacked
                return np.stack([1.0 - avg_pos, avg_pos], axis=1)
            return self._xgb.predict_proba(X_scaled)

        if not self._nets:
            raise RuntimeError("Classifier is not fitted yet. Call fit() first.")
        X_t = torch.from_numpy(X_scaled).float()
        probs = []
        with torch.no_grad():
            for net in self._nets:
                logits = net(X_t).squeeze(-1)
                probs.append(torch.sigmoid(logits).numpy())
        prob_pos = np.mean(np.stack(probs, axis=0), axis=0)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

