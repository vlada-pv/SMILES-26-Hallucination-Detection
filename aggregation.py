"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import os
import torch


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    mode = os.getenv("AGG_MODE", "last_token").lower()

    # Indices of real tokens (non-padding).
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    n_real = int(attention_mask.sum().item())

    if mode == "last_token":
        layer_ids_raw = os.getenv("AGG_LAYER_IDS", "").strip()
        if layer_ids_raw:
            # Example: AGG_LAYER_IDS="-1,-2,-4" -> concatenate last-token vectors from these layers.
            ids: list[int] = []
            for part in layer_ids_raw.split(","):
                part = part.strip()
                if not part:
                    continue
                ids.append(int(part))
            feats = [hidden_states[i, last_pos, :] for i in ids]
            return torch.cat(feats, dim=0)

        layer = hidden_states[-1]  # (seq_len, hidden_dim)
        return layer[last_pos]

    if mode == "midlate_answer_mean":
        # Middle-to-late layer pooling over an answer-window proxy.
        # Defaults are tuned for a 24-layer model: layers 12..20 inclusive.
        layer_ids_raw = os.getenv("AGG_MIDLATE_LAYER_IDS", "12,13,14,15,16,17,18,19,20")
        layer_ids: list[int] = []
        for part in layer_ids_raw.split(","):
            part = part.strip()
            if not part:
                continue
            layer_ids.append(int(part))

        if not layer_ids:
            layer_ids = [-1]

        # Keep valid ids only.
        valid_ids = [i for i in layer_ids if -hidden_states.size(0) <= i < hidden_states.size(0)]
        if not valid_ids:
            valid_ids = [-1]

        frac = float(os.getenv("AGG_ANSWER_WINDOW_FRAC", "0.4"))
        frac = min(max(frac, 0.05), 1.0)
        window = max(1, int(round(n_real * frac)))
        token_start = max(0, last_pos - window + 1)
        token_slice = slice(token_start, last_pos + 1)

        layer_vecs = []
        for i in valid_ids:
            # Mean over selected token window for each layer.
            layer_vecs.append(hidden_states[i, token_slice, :].mean(dim=0))

        pooled = torch.stack(layer_vecs, dim=0).mean(dim=0)

        # Optional L2 normalization to stabilize downstream tree/logistic models.
        if os.getenv("AGG_L2_NORM", "1").strip().lower() in {"1", "true", "yes"}:
            pooled = torch.nn.functional.normalize(pooled.unsqueeze(0), p=2, dim=1).squeeze(0)
        return pooled

    # Layer selection: last K transformer layers (excluding embeddings at index 0 is not required here).
    k_layers = int(os.getenv("AGG_LAST_K_LAYERS", "8"))
    k_layers = max(1, min(k_layers, hidden_states.size(0)))
    layers = hidden_states[-k_layers:]  # (k_layers, seq_len, hidden_dim)

    # Token selection: full real sequence or "answer window" proxy (last fraction).
    if mode in {"mean_last_layers", "delta_last_layers"}:
        token_start = 0
    elif mode in {"mean_answer_window", "delta_answer_window"}:
        frac = float(os.getenv("AGG_ANSWER_WINDOW_FRAC", "0.4"))  # last 40% of real tokens
        frac = min(max(frac, 0.05), 1.0)
        window = max(1, int(round(n_real * frac)))
        token_start = max(0, last_pos - window + 1)
    else:
        # Unknown mode -> fall back safely.
        layer = hidden_states[-1]
        return layer[last_pos]

    token_slice = slice(token_start, last_pos + 1)
    window = layers[:, token_slice, :]  # (k_layers, win_len, hidden_dim)

    if mode.startswith("mean"):
        # Mean pooling over tokens, then mean over layers.
        pooled = window.mean(dim=1).mean(dim=0)  # (hidden_dim,)
        return pooled

    # Residual dynamics: deltas between consecutive layers.
    # delta[l] = h[l] - h[l-1], for selected late layers.
    deltas = layers[1:] - layers[:-1]  # (k_layers-1, seq_len, hidden_dim)
    deltas_w = deltas[:, token_slice, :]  # (k_layers-1, win_len, hidden_dim)

    # Use mean update direction as representation of "how the model changes" late in the stack.
    pooled_delta = deltas_w.mean(dim=1).mean(dim=0)  # (hidden_dim,)
    return pooled_delta
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # Placeholder: returns an empty tensor (no geometric features).
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
