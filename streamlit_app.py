import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# Load the trained model
model = joblib.load('rf_model.pkl')

# Set Streamlit page settings
st.set_page_config(page_title='PulseAI - Diabetes Risk Predictor', layout='centered')
st.title('🧠 PulseAI')
st.subheader('Check your diabetes risk with smart ML insights')

# --- Define the input fields ---
pregnancies = st.number_input('Pregnancies', 0, 20, step=1)
glucose = st.number_input('Glucose', 0, 200)
skin = st.number_input('Skin Thickness', 0, 100)
bmi = st.number_input('BMI', 0.0, 70.0)
age = st.number_input('Age', 1, 120)

# Predict button
if st.button('🔍 Predict Risk'):
    # Prepare input DataFrame
    input_df = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'SkinThickness': skin,
        'BMI': bmi,
        'Age': age
    }])

    # Make prediction and get probability
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    # Display result
    if prediction == 1:
        st.error("⚠ High risk of diabetes")
        st.markdown(f"🧪 *Model confidence:* {round(proba * 100, 2)}%")
    else:
        st.success("✅ Low risk prediction")
        st.markdown(f"🧪 *Model confidence:* {round((1 - proba) * 100, 2)}%")

    # SHAP explainability
    st.subheader("🧠 Why this prediction?")

    try:
        # Use TreeExplainer for RandomForest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        st_shap(
            shap.plots.bar(
                shap.Explanation(
                    values=shap_values[0][0],  # SHAP values for the first (and only) sample
                    base_values=explainer.expected_value[0],
                    data=input_df.iloc[0],
                    feature_names=input_df.columns.tolist()
                )
            ),
            height=300
        )

    except Exception as e:
        st.warning(f"SHAP plot failed: {e}")