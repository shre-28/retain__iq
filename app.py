import os
import joblib
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="RetainIQ", page_icon="📊", layout="wide")

# 2. Strict Custom CSS for Layout Consistency
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }
    
    body {
        background-color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Input Box Styles */
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
    /* Submit Button */
    div[data-testid="stForm"] button {
        background-color: #0f172a;
        color: #ffffff;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        float: right;
        border: none;
        margin-top: 10px;
    }
    div[data-testid="stForm"] button:hover {
        background-color: #1e293b;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Loading
MODEL_PATH = os.path.join('notebooks - Copy', 'xgboost_churn_model.pkl')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'xgboost_churn_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# 4. Top Header Navigation
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; border-bottom: 1px solid #e2e8f0; padding-bottom: 12px;">
    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="background-color: #0f172a; color: #ffffff; font-weight: 800; font-size: 16px; width: 32px; height: 32px; border-radius: 6px; display: flex; align-items: center; justify-content: center;">
            R
        </div>
        <div>
            <div style="font-size: 15px; font-weight: 800; color: #0f172a; line-height: 1;">RetainIQ</div>
            <div style="font-size: 10px; color: #64748b; font-weight: 500;">Customer Intelligence</div>
        </div>
    </div>
    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 20px; padding: 4px 12px; display: flex; align-items: center; gap: 6px;">
        <span style="height: 7px; width: 7px; background-color: #22c55e; border-radius: 50%; display: inline-block;"></span>
        <span style="font-size: 11px; font-weight: 600; color: #64748b;">Model Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 5. Banner Section (Native Streamlit Column Split for Responsiveness)
head_col1, head_col2 = st.columns([3, 1])

with head_col1:
    st.markdown("""
    <div style="font-size: 11px; font-weight: 700; color: #2563eb; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 6px;">
        CUSTOMER RETENTION ANALYTICS
    </div>
    <h1 style="font-size: 28px; font-weight: 800; color: #0f172a; margin: 0; line-height: 1.2;">
        Understand customer <span style="color: #2563eb;">churn risk</span> before it happens.
    </h1>
    <p style="color: #64748b; font-size: 13px; margin-top: 6px;">
        Enter customer information to generate an AI-powered churn risk assessment.
    </p>
    """, unsafe_allow_html=True)

with head_col2:
    st.markdown("""
    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
        <div style="font-size: 9px; font-weight: 700; color: #64748b; letter-spacing: 0.5px;">ACTIVE MODEL</div>
        <div style="font-size: 16px; font-weight: 800; color: #0f172a; margin: 2px 0 8px 0;">XGBoost</div>
        <div style="display: flex; justify-content: space-between;">
            <div>
                <div style="font-size: 12px; font-weight: 700; color: #2563eb;">86.87%</div>
                <div style="font-size: 9px; color: #94a3b8;">Test Accuracy</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 12px; font-weight: 700; color: #2563eb;">ML</div>
                <div style="font-size: 9px; color: #94a3b8;">Classification</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Form Title
st.markdown("""
<div>
    <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0;">Customer Profile</h3>
    <p style="color: #64748b; font-size: 12px; margin-top: 2px; margin-bottom: 12px;">Provide the customer's account information for analysis.</p>
</div>
""", unsafe_allow_html=True)

# 7. Main Input Form
with st.form("churn_form"):
    st.markdown("<p style='font-weight: 700; font-size: 12px; color: #334155; margin-bottom: 4px;'>Personal Information</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        age = st.number_input("Age", 18, 100, 40)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])

    st.markdown("<p style='font-weight: 700; font-size: 12px; color: #334155; margin-top: 10px; margin-bottom: 4px;'>Account Information</p>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        tenure = st.number_input("Tenure (Years)", 0, 10, 5)
        num_products = st.number_input("Number of Products", 1, 4, 2)
    with col4:
        balance = st.number_input("Account Balance ($)", 0.0, 500000.0, 100000.0)
        estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 500000.0, 100000.0)

    st.markdown("<p style='font-weight: 700; font-size: 12px; color: #334155; margin-top: 10px; margin-bottom: 4px;'>Customer Engagement</p>", unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        has_credit_card = st.selectbox("Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    with col6:
        is_active_member = st.selectbox("Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    submit_button = st.form_submit_button("Analyze Churn Risk →")

# 8. Assessment Display
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
            title_color = "#dc2626"
            bar_color = "#ef4444"
            status_text = "Customer has a HIGH churn risk"
            badge_text = "HIGH RISK"
            badge_bg = "#fee2e2"
            badge_color = "#991b1b"
            rec_text = "Recommendation: Consider applying targeted retention incentives."
        else:
            title_color = "#16a34a"
            bar_color = "#22c55e"
            status_text = "Customer has a LOW churn risk"
            badge_text = "LOW RISK"
            badge_bg = "#dcfce7"
            badge_color = "#166534"
            rec_text = "Recommendation: This customer currently shows a lower likelihood of leaving."

        st.markdown(f"""
        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-top: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
            <div style="font-size: 10px; font-weight: 700; color: #2563eb; letter-spacing: 0.5px; margin-bottom: 6px;">RISK ASSESSMENT</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2 style="color: {title_color}; font-size: 20px; font-weight: 700; margin: 0;">{status_text}</h2>
                <span style="font-size: 32px; font-weight: 800; color: #0f172a;">{churn_prob}%</span>
            </div>
            <div style="background-color: #f1f5f9; border-radius: 999px; height: 8px; width: 100%; margin: 12px 0 10px 0; overflow: hidden;">
                <div style="background-color: {bar_color}; width: {churn_prob}%; height: 100%; border-radius: 999px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #64748b; font-size: 12px;">{rec_text}</span>
                <span style="background-color: {badge_bg}; color: {badge_color}; font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 999px;">{badge_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
