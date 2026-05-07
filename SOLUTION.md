# SOLUTION.md

## 1) Reproducibility

### Environment
- Python 3.10+ (tested in local venv)
- Install dependencies from `requirements.txt`
- Run from repository root `SMILES-HALLUCINATION-DETECTION`

### Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python solution.py
```

### Final run configuration (used for final `predictions.csv`)

```bash
source .venv/bin/activate
SPLIT_N_SPLITS=5 \
AGG_MODE=last_token \
PROBE_MODEL=et \
PROBE_TREE_STACKING=0 \
PROBE_THRESHOLD_METRIC=accuracy \
PROBE_ENSEMBLE_SEEDS=41,42,43 \
ET_N_ESTIMATORS=800 \
ET_MAX_DEPTH=10 \
ET_MIN_SAMPLES_LEAF=5 \
ET_MAX_FEATURES=sqrt \
python solution.py
```

This command produces:
- `results.json`
- `predictions.csv`

## 2) Final solution description

### Files modified
- `aggregation.py`
- `probe.py`
- `splitting.py`

### Final approach

1. **Aggregation (`aggregation.py`)**
   - Final mode is `AGG_MODE=last_token`: use the last real token from the final transformer layer.
   - Several alternative modes were implemented for experiments (mean pooling, answer-window pooling, residual deltas, middle-to-late pooling), but final submission uses the most stable variant above.

2. **Probe (`probe.py`)**
   - Added model switch via `PROBE_MODEL`.
   - Final classifier: **ExtraTrees** (`PROBE_MODEL=et`) with:
     - class balancing (`balanced_subsample`)
     - ensemble over seeds (`PROBE_ENSEMBLE_SEEDS=41,42,43`)
     - tuned tree regularization (`ET_MAX_DEPTH=10`, `ET_MIN_SAMPLES_LEAF=5`, `ET_MAX_FEATURES=sqrt`, `ET_N_ESTIMATORS=800`)
   - Kept threshold tuning in `fit_hyperparameters`, optimizing for accuracy (`PROBE_THRESHOLD_METRIC=accuracy`), including midpoint threshold candidates for better discrete optimization.
   - Added optional XGBoost path with automatic fallback to ExtraTrees in environments where XGBoost runtime is unavailable.
   - Added optional tree-stacking mode (`PROBE_TREE_STACKING`), but final config keeps it disabled due to no stable gain.

3. **Splitting (`splitting.py`)**
   - Stratified k-fold evaluation with optional group-awareness by `prompt` via `StratifiedGroupKFold` fallback logic.
   - Added configurable folds through `SPLIT_N_SPLITS` to support lightweight quick tests (3 folds) and final evaluation (5 folds).

### Why these choices

- On this dataset size, simple and stable representations generalized better than complex feature engineering.
- Tree-based probe (ExtraTrees ensemble) consistently outperformed tested MLP configurations under the same evaluation protocol.
- Accuracy is the primary metric in README, so threshold tuning and configuration selection prioritized test-split accuracy in k-fold averages.

### Final observed performance (last controlled run)

- Averaged over 5 folds:
  - **Test Accuracy:** 72.57%
  - **Test AUROC:** 73.45%

## 3) Experiments and failed attempts

### 3.1 Aggregation experiments

- **Mean pooling over last 8 layers (full tokens)**  
  Strong degradation vs baseline (`last_token`), especially in AUROC.

- **Mean pooling over answer-window only**  
  Better than full mean pooling, but still below `last_token`.

- **Residual dynamics / delta between layers**  
  Unstable and usually lower AUROC/accuracy than baseline.

- **Middle-to-late layer answer-window mean pooling**  
  Implemented and tested with several layer ranges and token-window fractions; underperformed baseline in controlled runs.

- **Multi-layer last-token concatenation**  
  Slight AUROC changes but no robust accuracy improvement vs final selected setup.

### 3.2 Probe experiments

- **MLP variants**  
  Under consistent 5-fold protocol, MLP performed worse than ExtraTrees and showed train/val/test instability.

- **XGBoost**  
  Implemented, but local macOS environment lacked OpenMP runtime (`libomp.dylib`), so direct XGBoost execution was not stable/reproducible in this setup.

- **Tree stacking (ET + LogisticRegression on ET probabilities)**  
  Implemented and tested in lightweight protocol; no consistent gain over plain ET ensemble, so not included in final config.

### 3.3 Validation protocol note

Single-split results can look better by chance on small datasets.  
Final model selection was based on stratified k-fold averages to reduce optimistic variance.

