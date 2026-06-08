import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
# ==========================
# CONFIG
# ==========================

API_URL = "https://churn-predictor-ytzm.onrender.com/predict"

st.set_page_config(
    page_title="AI Churn Predictor",
    page_icon="📊",
    layout="wide"
)

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 1rem;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.hero {
    background: linear-gradient(90deg,#0066ff,#00c6ff);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    color: white;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================

st.sidebar.markdown("## 🤖 AI Churn Platform")
st.sidebar.markdown(
    "Customer Retention Analytics"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Prediction",
        "📜 History",
        "📈 Analytics",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

try:
    health = requests.get(
        "https://churn-predictor-ytzm.onrender.com/health",
        timeout=10
    )

    if health.status_code == 200:

        h = health.json()

        st.sidebar.success(
            f"🟢 {h['champion_model']} Online"
        )

except:
    st.sidebar.error("🔴 API Offline")

# ==========================
# HERO
# ==========================

if page == "🏠 Prediction":

    st.markdown("""
    <div class="hero">
        <h1>📊 AI Customer Churn Predictor</h1>
        <h4>
        Predict customer retention risk using
        XGBoost + LightGBM Models
        </h4>
        <p>
        FastAPI • Streamlit • Docker • MLflow • DVC • CI/CD
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            0,
            72,
            12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with col2:

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    st.divider()

    col3, col4 = st.columns(2)

    with col3:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No"
            ]
        )

    with col4:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.35
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=150.30
    )

    # ==========================
    # PREDICT BUTTON
    # ==========================

    if st.button("🚀 Predict Churn Risk"):

        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        with st.spinner("Analyzing customer..."):

            response = requests.post(
                API_URL,
                json=payload
            )

        if response.status_code == 200:

            result = response.json()

            st.session_state.history.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Probability (%)": round(result["churn_probability"] * 100, 2),
                "Risk": result["risk_level"],
                "Prediction": result["churn_label"],
                "Model": result["model_used"]
            })

            probability = result["churn_probability"]

            st.success("Prediction Completed")

            # ==========================
            # KPI CARDS
            # ==========================

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Churn Probability",
                    f"{probability:.2%}"
                )

            with c2:
                st.metric(
                    "Risk Level",
                    result["risk_level"]
                )

            with c3:
                st.metric(
                    "Model",
                    result["model_used"]
                )

            # ==========================
            # GAUGE
            # ==========================

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Churn Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 30], "color": "green"},
                        {"range": [30, 70], "color": "orange"},
                        {"range": [70, 100], "color": "red"}
                    ]
                }
            ))

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ==========================
            # RISK BADGE
            # ==========================

            if probability < 0.3:
                st.success("🟢 LOW RISK CUSTOMER")

            elif probability < 0.7:
                st.warning("🟠 MEDIUM RISK CUSTOMER")

            else:
                st.error("🔴 HIGH RISK CUSTOMER")

            # ==========================
            # EXPLANATIONS
            # ==========================

            st.subheader("Why this prediction?")

            reasons = []

            if tenure < 12:
                reasons.append(
                    "Short customer tenure"
                )

            if monthly_charges > 80:
                reasons.append(
                    "High monthly charges"
                )

            if contract == "Month-to-month":
                reasons.append(
                    "Month-to-month contract"
                )

            if tech_support == "No":
                reasons.append(
                    "No tech support"
                )

            if online_security == "No":
                reasons.append(
                    "No online security"
                )

            if len(reasons) == 0:
                reasons.append(
                    "Customer profile appears stable"
                )

            for reason in reasons:
                st.write(f"✅ {reason}")

        else:
            st.error(response.text)

# ==========================
# HISTORY PAGE
# ==========================

elif page == "📜 History":

    st.title("📜 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions made yet.")
    else:

        df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(index=False)

        st.download_button(
            "⬇ Download History CSV",
            csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()

# ==========================
# ANALYTICS PAGE
# ==========================

elif page == "📈 Analytics":

    st.title("📈 Analytics Dashboard")

    st.metric(
        "Total Predictions Made",
        len(st.session_state.history)
    )

    if len(st.session_state.history) > 0:

        df = pd.DataFrame(st.session_state.history)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predictions",
            len(df)
        )

        c2.metric(
            "High Risk",
            len(df[df["Risk"]=="HIGH"])
        )

        c3.metric(
            "Average Probability",
            f"{df['Probability (%)'].mean():.1f}%"
        )

        fig = go.Figure()

        fig.add_histogram(
            x=df["Probability (%)"]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        risk_counts = df["Risk"].value_counts()

        pie = go.Figure(
            data=[
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.4
                )
            ]
        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

    else:

        st.info(
            "No prediction history available."
        )

# ==========================
# ABOUT PAGE
# ==========================

else:

    st.title("ℹ️ About")

    st.markdown("""
# AI Customer Churn Prediction Platform

### Built By
Yuvraaj M N

### Tech Stack

- XGBoost
- LightGBM
- FastAPI
- Streamlit
- Docker
- Redis
- MLflow
- DVC
- Evidently AI
- GitHub Actions
- Render

### Features

✅ Real-Time Prediction

✅ Model Monitoring

✅ CI/CD Pipeline

✅ A/B Testing

✅ REST APIs

✅ Docker Deployment

✅ Prediction History

✅ Interactive Dashboard

### Deployment

Backend:
https://churn-predictor-ytzm.onrender.com

API Docs:
https://churn-predictor-ytzm.onrender.com/docs
""")

st.markdown("---")

st.caption(
    "Built with Streamlit, FastAPI, XGBoost, Docker and MLOps tools"
)