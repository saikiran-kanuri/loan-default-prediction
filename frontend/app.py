"""
frontend/app.py
Streamlit UI for the Loan Default Prediction API.

Collects the top ~9 SHAP-important features from the user, sends them to
the FastAPI /predict endpoint, and displays the prediction. Any feature
not shown here is intentionally left unset — predict.py fills it with
NaN and the pipeline's imputer handles it, so a partial form is a valid
request, not a missing-data error.
"""

import os
import requests
import streamlit as st

# Local dev default. When deployed, set API_URL as an environment variable
# pointing at the live Render URL (see Section 5.7 of the project doc).
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰")
st.title("💰 Loan Default Risk Predictor")
st.write(
    "Enter applicant details below. This uses the most influential "
    "features identified via SHAP analysis — not all 67 model inputs "
    "are required for a prediction."
)

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=13.5, step=0.1)
        term = st.selectbox("Loan Term", ["36 months", "60 months"])
        home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
        fico_range_low = st.number_input("FICO Score (low end of range)", min_value=300, max_value=850, value=680)
        dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, value=18.0, step=0.1)

    with col2:
        acc_open_past_24mths = st.number_input("Accounts Opened (past 24 months)", min_value=0, max_value=50, value=3)
        mort_acc = st.number_input("Number of Mortgage Accounts", min_value=0, max_value=20, value=1)
        emp_length = st.selectbox(
            "Employment Length",
            ["< 1 year", "1 year", "2 years", "3 years", "4 years",
             "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"],
            index=5,
        )
        addr_state = st.selectbox(
            "State",
            ["CA", "NY", "TX", "FL", "AZ", "IL", "OH", "PA", "GA", "NC", "Other"],
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "features": {
            "int_rate": int_rate,
            "term": term,
            "home_ownership": home_ownership,
            "fico_range_low": fico_range_low,
            "dti": dti,
            "acc_open_past_24mths": acc_open_past_24mths,
            "mort_acc": mort_acc,
            "emp_length": emp_length,
            "addr_state": addr_state,
        }
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        prediction = result["prediction"]
        probability_good = result["probability_good"]

        st.divider()
        if prediction == 1:
            st.success(f"✅ Likely to repay — confidence: {probability_good:.1%}")
        else:
            st.error(f"⚠️ Higher default risk — confidence of repayment: {probability_good:.1%}")

        st.progress(probability_good)

    except requests.exceptions.ConnectionError:
        st.error(
            "Couldn't reach the API. Make sure it's running locally with:\n\n"
            "`uvicorn api.main:app --reload`"
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e}")