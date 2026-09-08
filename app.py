import os
import joblib
import pandas as pd
import streamlit as st

# 1. Page Config
st.set_page_config(
    page_title="RetainIQ - Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# 2. Custom CSS Injection (Matches your screenshot UI style)
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #F8FAFC;
    }
    
    /* Header Container */
    .header-container {
        padding: 1.5rem 0;
        margin-bottom: 2rem;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
    }
    .header-title span {
        color: #2563EB;
    }
    .header-subtitle {
        color: #64748B;
        font-size: 1rem;
    }
    
    /* Input Section Cards */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Custom Card Box */
    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Submit Button */
    div[data-testid="stForm"] button {
        background-color: #0F172A;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        float: right;
        border: none;
    }
    div[data-testid="stForm"] button:hover {
        background-color: #1E293B;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Loading Logic
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

# 4. Top Header Banner
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("""
        <div class="header-container">
            <div class="header-title">Understand customer <span>churn risk</span> before it happens.</div>
            <div class="header-subtitle">Enter customer information to generate an AI-powered churn risk assessment.</div>
        </div>
    """, unsafe_allow_html=True)

with col_head2:
    st.info("**ACTIVE MODEL**  \n**XGBoost**  \n86.87% Test Accuracy")

# 5. Customer Profile Form
st.markdown("### Customer Profile")
st.caption("Provide the customer's account information for analysis.")

with st.form("churn_form"):
    
    st.markdown('<div class="section-title">Personal Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, help="Range: 300 – 850")
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])

    st.markdown('<div class="section-title">Account Information</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5, help="Years with the bank")
        num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
    with c4:
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=100000.0)
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=100000.0)

    st.markdown('<div class="section-title">Customer Engagement</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        has_credit_card = st.selectbox("Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    with c6:
        is_active_member = st.selectbox("Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("Analyze Churn Risk →")

# 6. Prediction Output Styling
if submit_button:
    if model is None:
        st.error("Model file loading error.")
    else:
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

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Risk Display Box
            with st.container():
                st.caption("RISK ASSESSMENT")
                res_col1, res_col2 = st.columns([3, 1])
                
                with res_col1:
                    if prediction == 1:
                        st.error("### Customer has a HIGH churn risk")
                        st.caption("Recommendation: Consider applying targeted retention incentives.")
                    else:
                        st.success("### Customer has a LOW churn risk")
                        st.caption("Recommendation: This customer currently shows a lower likelihood of leaving.")
                        
                with res_col2:
                    st.metric(label="Churn Risk", value=f"{churn_prob}%")
                
                st.progress(float(probability))

        except Exception as e:
            st.error(f"Prediction error: {e}")
