import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# 1. Page Config Setup
st.set_page_config(
    page_title="RetainIQ - Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# 2. Title & Description
st.title("📊 RetainIQ: Customer Churn Prediction")
st.write("Fill in the customer details below to predict churn probability.")

# 3. Model Loading Logic
MODEL_PATH = os.path.join('notebooks - Copy', 'xgboost_churn_model.pkl')

@st.cache_resource
def load_churn_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Model file not found at `{MODEL_PATH}`")
        return None

model = load_churn_model()

# 4. Input Form (Tumchya dataset validation inputs nusar adjust kara)
st.subheader("Customer Information")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        
    with col2:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    submit_button = st.form_submit_button(label="Predict Churn Risk")

# 5. Prediction Execution
if submit_button:
    if model is None:
        st.error("Model is not loaded properly. Please check your repository files.")
    else:
        # Prepare input data dictionary to match model features
        input_data = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Contract': contract
        }
        
        input_df = pd.DataFrame([input_data])
        
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
            
            st.markdown("---")
            if prediction == 1:
                st.error("⚠️ **High Risk of Churn!**")
                if probability is not None:
                    st.write(f"Churn Probability: **{probability * 100:.2f}%**")
            else:
                st.success("✅ **Customer is Likely to Stay (Retained)**")
                if probability is not None:
                    st.write(f"Retention Probability: **{(1 - probability) * 100:.2f}%**")
                    
        except Exception as e:
            st.error(f"Prediction failed: {e}")
