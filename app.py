import os
import joblib
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="RetainIQ - Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 RetainIQ: Customer Churn Prediction")
st.write("Enter customer details to predict churn risk.")

# 2. Model Path Logic (Checking both standard locations)
MODEL_PATH = os.path.join('notebooks - Copy', 'xgboost_churn_model.pkl')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'xgboost_churn_model.pkl'

@st.cache_resource
def load_churn_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Model file not found at `{MODEL_PATH}`")
        return None

model = load_churn_model()

# 3. Form UI (Matching exact original features)
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)

    with col2:
        balance = st.number_input("Balance ($)", min_value=0.0, value=60000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_credit_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0)

    submit_button = st.form_submit_button("Predict Churn Risk")

# 4. Prediction Logic
if submit_button:
    if model is None:
        st.error("Model isn't loaded properly. Verify repo files.")
    else:
        # Exact DataFrame keys matching training schema
        customer_data = pd.DataFrame([{
            "CreditScore": float(credit_score),
            "Geography": geography,
            "Gender": gender,
            "Age": float(age),
            "Tenure": float(tenure),
            "Balance": float(balance),
            "NumOfProducts": float(num_products),
            "HasCrCard": float(has_credit_card),
            "IsActiveMember": float(is_active_member),
            "EstimatedSalary": float(estimated_salary)
        }])

        try:
            prediction = model.predict(customer_data)[0]
            probability = model.predict_proba(customer_data)[0][1]
            churn_prob = round(probability * 100, 2)

            st.markdown("---")
            if prediction == 1:
                st.error(f"⚠️ **Customer is likely to churn**")
                st.write(f"Churn Probability: **{churn_prob}%**")
            else:
                st.success(f"✅ **Customer is likely to stay**")
                st.write(f"Retention Probability: **{round(100 - churn_prob, 2)}%**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
