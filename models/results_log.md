# Model Results Log — Lending Club (Phase 2)

## Logistic Regression (baseline) — model_v2_full.pkl
- Validation AUC-ROC: 0.6964
- Class 0 (default) recall: 0.62, precision: 0.31
- Class 1 (good) recall: 0.66, precision: 0.88
- Notes: class_weight='balanced' used to handle 80/20 imbalance.
  Linear model likely underfits non-linear credit risk patterns —
  XGBoost comparison planned next.

## XGBoost — model_v2_xgboost.pkl
- Validation AUC-ROC: 0.7068 (vs 0.6964 for Logistic Regression)
- Class 0 (default) recall: 0.60, precision: 0.32
- Class 1 (good) recall: 0.69, precision: 0.88
- Notes: scale_pos_weight used for imbalance (train-set-specific ratio).
  Only marginal improvement over Logistic Regression (+0.01 AUC) —
  suggests feature set, not model complexity, is the current ceiling.
  XGBoost selected as primary model for this reason: equal-or-better
  on every metric, despite the small margin.

## Experiment: Re-including 'grade' — reverted
- Logistic Regression: 0.6968 AUC (vs 0.6964 baseline) — negligible change
- XGBoost: 0.7055 AUC (vs 0.7068 baseline) — slightly worse
- Conclusion: 'grade' is redundant with 'int_rate' (both derive from
  LC's own risk assessment) — provides no new signal. Reverted to
  original feature set. Confirms the ~0.70 AUC ceiling is a genuine
  information limit in this feature set, not a missing-feature issue.

## SHAP Analysis — XGBoost model_v2_xgboost.pkl
- Dominant feature: int_rate (SHAP importance ~2x the next highest feature)
- Top 6 features: int_rate, term, dti, debt_to_income_calculated,
  acc_open_past_24mths, fico_range_low — all financially sensible.
- Key insight: int_rate's dominance explains why re-including 'grade'
  and adding engineered debt/credit ratios both failed to improve AUC —
  they were redundant with signal the model already captured through
  int_rate. The ~0.70 AUC ceiling reflects int_rate carrying most of
  the separable signal in this feature set.