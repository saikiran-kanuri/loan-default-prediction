# Loan Default Risk Prediction

An end-to-end machine learning system that predicts whether a loan applicant is likely to repay or default, trained on real Lending Club loan data. Built and deployed as a live, working web application — not just a notebook.

**Live demo:** App: https://loan-default-prediction-rf2efsgltxzdrmumqcljur.streamlit.app | API docs: https://loan-default-prediction-276p.onrender.com/docs

## Problem

Lenders need to estimate default risk before approving a loan. This project trains a classifier on ~67,000 resolved Lending Club loans to predict repayment outcome, then serves that model through a REST API and a simple web interface so the prediction can actually be used, not just evaluated in a notebook.

## Approach

- **Data**: Lending Club loan dataset, filtered to resolved loans only (repaid or charged off), ~67K rows, roughly 80/20 class balance.
- **Feature engineering**: 67 features retained after three deliberate filtering passes — dropping columns with >50% missing values, columns that leak post-outcome information (e.g. fields only populated after default), and Lending-Club-internal/redundant fields. `earliest_cr_line` is converted into `credit_history_months`.
- **Models compared**: Logistic Regression (AUC 0.70) and XGBoost (AUC 0.71), both trained as complete `sklearn.Pipeline` objects — preprocessing and model bundled into a single artifact, so there's no risk of train/inference preprocessing drift.
- **Class imbalance**: handled via `scale_pos_weight` (XGBoost) rather than naive resampling, and evaluated on AUC-ROC/precision/recall/F1 rather than accuracy alone, since a model can hit 95%+ accuracy by always predicting "repaid" on this imbalanced data.
- **Explainability**: SHAP analysis confirms `int_rate` is the dominant predictor by a wide margin — sensible, since interest rate already encodes the lender's own risk assessment at origination. This also explains the AUC ceiling: most of the usable signal is already priced into `int_rate`, leaving limited independent signal in the remaining features.

## Results

| Model | Validation AUC-ROC |
|---|---|
| Logistic Regression | 0.696 |
| XGBoost | 0.709 |

An AUC around 0.70–0.71 is the genuine, expected ceiling for this feature set — not a modeling shortfall. Two experiments confirmed this: re-including `grade` (redundant with `int_rate`, negligible effect) and adding engineered debt ratios (redundant with existing `dti`/`revol_util`, negligible effect). Both are logged in `models/results_log.md`.

## Architecture

```
Raw CSV → data_loader.py → features.py → train.py → models/model_v2_xgboost.pkl
                                                              |
User (browser) → Streamlit (frontend/app.py) → FastAPI (api/main.py) → predict.py → prediction
```

The trained pipeline bundles preprocessing and model together, so the API only ever loads one artifact and calls `.predict()` — no separate preprocessing step to keep in sync at inference time.

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Streamlit, Docker, Render.

## Running locally

```bash
# clone and set up
git clone https://github.com/saikiran-kanuri/loan-default-prediction.git
cd loan-default-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# train the model
python3 -m src.train --model xgboost

# start the API
uvicorn api.main:app --reload
# visit http://127.0.0.1:8000/docs

# in a separate terminal, start the frontend
streamlit run frontend/app.py
```

## Running with Docker

```bash
docker build -t loan-default-api .
docker run -p 8000:8000 loan-default-api
```

## Project structure

```
src/            data loading, feature engineering, training, prediction
api/            FastAPI backend
frontend/       Streamlit UI
models/         trained pipeline, SHAP plots, experiment log
notebooks/      exploratory analysis only (not used in the deployed app)
```

## What I'd improve with more time

- Swap in the Home Credit Default Risk dataset (multiple relational tables) to demonstrate feature engineering across joins, not just a single flat CSV.
- Add authentication and rate limiting to the API before treating it as anything beyond a portfolio demo.
- Add automated tests for the FastAPI endpoints, not just `predict.py`.
- Explore calibration of the predicted probabilities, since AUC alone doesn't guarantee well-calibrated confidence scores.