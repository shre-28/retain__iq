import os
import joblib
import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(page_title="RetainIQ", page_icon="📊", layout="wide")

# Hide Streamlit Chrome/Footer
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 0rem;}
    
    /* Clean Body Base */
    body {
        background-color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
MODEL_PATH = os.path.join('notebooks - Copy', 'xgboost_churn_model.pkl')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'xgboost_churn_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# Header HTML
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 25px;">
    <div>
        <h1 style="font-size: 28px; font-weight: 800; color: #0f172a; margin: 0;">
            Understand customer <span style="color: #2563eb;">churn risk</span> before it happens.
        </h1>
        <p style="color: #64748b; font-size: 14px; margin-top: 5px;">
            Enter customer information to generate an AI-powered churn risk assessment.
        </p>
    </div>
    <div style="background-color: #eff6ff; border: 1px solid #dbeafe; border-radius: 8px; padding: 12px 20px; text-align: left;">
        <span style="font-size: 11px; font-weight: 700; color: #3b82f6; letter-spacing: 0.5px;">ACTIVE MODEL</span><br>
        <span style="font-size: 15px; font-weight: 700; color: #1e293b;">XGBoost</span><br>
        <span style="font-size: 12px; color: #64748b;">86.87% Test Accuracy</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Form Title
st.markdown("""
<div style="margin-bottom: 15px;">
    <h3 style="font-size: 18px; font-weight: 700; color: #0f172a; margin: 0;">Customer Profile</h3>
    <p style="color: #64748b; font-size: 13px; margin-top: 2px;">Provide the customer's account information for analysis.</p>
</div>
""", unsafe_allow_html=True)

# Form Layout
with st.form("churn_form"):
    
    st.markdown("<p style='font-weight: 700; font-size: 13px; color: #334155; margin-bottom: 5px;'>Personal Information</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        age = st.number_input("Age", 18, 100, 40)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])

    st.markdown("<p style='font-weight: 700; font-size: 13px; color: #334155; margin-top: 15px; margin-bottom: 5px;'>Account Information</p>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        tenure = st.number_input("Tenure (Years)", 0, 10, 5)
        num_products = st.number_input("Number of Products", 1, 4, 2)
    with col4:
        balance = st.number_input("Account Balance ($)", 0.0, 500000.0, 100000.0)
        estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 500000.0, 100000.0)

    st.markdown("<p style='font-weight: 700; font-size: 13px; color: #334155; margin-top: 15px; margin-bottom: 5px;'>Customer Engagement</p>", unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        has_credit_card = st.selectbox("Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    with col6:
        is_active_member = st.selectbox("Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    submit_button = st.form_submit_button("Analyze Churn Risk →")

# Output Processing (Custom Pixel HTML Result)
if submit_button:
    if model is not None:
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

        prediction = model.predict(customer_data)[0]
        probability = model.predict_proba(customer_data)[0][1]
        churn_prob = round(probability * 100, 2)

        if prediction == 1:
            bg_color = "#fef2f2"
            border_color = "#fecaca"
            title_color = "#dc2626"
            bar_color = "#ef4444"
            status_text = "Customer has a HIGH churn risk"
            badge_text = "HIGH RISK"
            badge_bg = "#fee2e2"
            badge_color = "#991b1b"
            rec_text = "Recommendation: Consider applying targeted retention incentives."
        else:
            bg_color = "#f0fdf4"
            border_color = "#bbf7d0"
            title_color = "#16a34a"
            bar_color = "#22c55e"
            status_text = "Customer has a LOW churn risk"
            badge_text = "LOW RISK"
            badge_bg = "#dcfce7"
            badge_color = "#166534"
            rec_text = "Recommendation: This customer currently shows a lower likelihood of leaving."

        # Pure HTML Output Box like original design
        st.markdown(f"""
        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 25px; margin-top: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 11px; font-weight: 700; color: #2563eb; letter-spacing: 0.5px; margin-bottom: 8px;">RISK ASSESSMENT</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2 style="color: {title_color}; font-size: 22px; font-weight: 700; margin: 0;">{status_text}</h2>
                <span style="font-size: 36px; font-weight: 800; color: #0f172a;">{churn_prob}%</span>
            </div>
            <div style="background-color: #f1f5f9; border-radius: 999px; height: 8px; width: 100%; margin: 15px 0 12px 0; overflow: hidden;">
                <div style="background-color: {bar_color}; width: {churn_prob}%; height: 100%; border-radius: 999px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #64748b; font-size: 13px;">{rec_text}</span>
                <span style="background-color: {badge_bg}; color: {badge_color}; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 999px;">{badge_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
