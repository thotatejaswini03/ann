import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# LOAD MODEL FILES
# ==================================================

model = tf.keras.models.load_model(
    "models/ann_model.keras"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

training_columns = joblib.load(
    "models/training_columns.pkl"
)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("ℹ️ Model Info")

    st.markdown("---")

    st.write("**Model:** Artificial Neural Network")

    st.write("**Architecture:**")
    st.code("""
Input Layer
↓
64 Neurons (ReLU)
↓
32 Neurons (ReLU)
↓
16 Neurons (ReLU)
↓
1 Neuron (Sigmoid)
""")

    st.write("**Optimizer:** Adam")
    st.write("**Loss:** Binary Crossentropy")

# ==================================================
# TITLE
# ==================================================

st.title("📊 Customer Churn Prediction System")

st.markdown(
    "Predict whether a telecom customer is likely to leave the service."
)

st.markdown("---")

# ==================================================
# INPUT SECTIONS
# ==================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("👤 Customer Information")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        24
    )

with col2:

    st.subheader("📞 Service Information")

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes"]
    )

st.markdown("---")

col3, col4 = st.columns(2)

with col3:

    st.subheader("📄 Account Information")

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col4:

    st.subheader("💰 Financial Information")

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1500.0
    )

# ==================================================
# PREDICT BUTTON
# ==================================================

if st.button("🚀 Predict Churn"):

    input_data = {
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    temp_df = pd.DataFrame([input_data])

    # create all training columns
    final_df = pd.DataFrame(
        0,
        index=[0],
        columns=training_columns
    )

    for col in temp_df.columns:
        if col in final_df.columns:
            final_df[col] = temp_df[col]

    # Gender
    if "gender_Male" in final_df.columns:
        final_df["gender_Male"] = (
            1 if gender == "Male" else 0
        )

    # Partner
    if "Partner_Yes" in final_df.columns:
        final_df["Partner_Yes"] = (
            1 if partner == "Yes" else 0
        )

    # Dependents
    if "Dependents_Yes" in final_df.columns:
        final_df["Dependents_Yes"] = (
            1 if dependents == "Yes" else 0
        )

    # Phone Service
    if "PhoneService_Yes" in final_df.columns:
        final_df["PhoneService_Yes"] = (
            1 if phone_service == "Yes" else 0
        )

    # Internet Service
    if (
        internet_service == "Fiber optic"
        and
        "InternetService_Fiber optic"
        in final_df.columns
    ):
        final_df[
            "InternetService_Fiber optic"
        ] = 1

    elif (
        internet_service == "No"
        and
        "InternetService_No"
        in final_df.columns
    ):
        final_df[
            "InternetService_No"
        ] = 1

    # Online Security
    if "OnlineSecurity_Yes" in final_df.columns:
        final_df["OnlineSecurity_Yes"] = (
            1 if online_security == "Yes" else 0
        )

    # Streaming TV
    if "StreamingTV_Yes" in final_df.columns:
        final_df["StreamingTV_Yes"] = (
            1 if streaming_tv == "Yes" else 0
        )

    # Contract
    if (
        contract == "One year"
        and
        "Contract_One year"
        in final_df.columns
    ):
        final_df[
            "Contract_One year"
        ] = 1

    elif (
        contract == "Two year"
        and
        "Contract_Two year"
        in final_df.columns
    ):
        final_df[
            "Contract_Two year"
        ] = 1

    # Paperless Billing
    if "PaperlessBilling_Yes" in final_df.columns:
        final_df[
            "PaperlessBilling_Yes"
        ] = (
            1 if paperless_billing == "Yes"
            else 0
        )

    # Payment Method
    payment_col = (
        f"PaymentMethod_{payment_method}"
    )

    if payment_col in final_df.columns:
        final_df[payment_col] = 1

    # Scale
    scaled_input = scaler.transform(
        final_df
    )

    prediction = model.predict(
        scaled_input,
        verbose=0
    )

    probability = float(
        prediction[0][0]
    )

    st.markdown("---")

    st.subheader("📈 Prediction Results")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

    with metric_col2:

        if probability < 0.30:
            st.success(
                "🟢 Low Risk Customer"
            )

        elif probability < 0.70:
            st.warning(
                "🟡 Medium Risk Customer"
            )

        else:
            st.error(
                "🔴 High Risk Customer"
            )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={
                "text":
                "Customer Churn Risk (%)"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                }
            }
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )