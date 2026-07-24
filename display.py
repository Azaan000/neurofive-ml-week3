import streamlit as st
import pandas as pd
import joblib


# Load model
tree_model = joblib.load("tree_model.pkl")


st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="centered"
)


st.title("📊 Customer Churn Prediction App")

st.write(
    "Enter customer details to predict whether the customer is likely to churn."
)


# Inputs

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


monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)


total = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=500.0
)


contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)



if st.button("Predict Churn"):


    # Create input dataframe
    data = {

        "SeniorCitizen":[senior],

        "tenure":[tenure],

        "MonthlyCharges":[monthly],

        "TotalCharges":[total],

        "gender":[gender],

        "Partner":[partner],

        "Dependents":[dependents],

        "Contract":[contract],

        "PaymentMethod":[payment]

    }


    input_df = pd.DataFrame(data)


    # One hot encoding
    input_df = pd.get_dummies(
        input_df,
        drop_first=True
    )


    # Add missing columns required by tree model
    model_features = tree_model.feature_names_in_


    input_df = input_df.reindex(
        columns=model_features,
        fill_value=0
    )


    prediction = tree_model.predict(input_df)


    if prediction[0] == 1:

        st.error(
            "⚠️ Customer is likely to churn"
        )

    else:

        st.success(
            "✅ Customer is likely to stay"
        )