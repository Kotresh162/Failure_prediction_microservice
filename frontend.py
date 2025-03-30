import streamlit as st
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("CSV Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df)

    if st.button("Get Predictions"):
        predictions = []
        for _, row in df.iterrows():
            features = row.tolist()
            try:
                response = requests.post(API_URL, json={"features": features})
                if response.status_code == 200:
                    predictions.append(response.json())
                else:
                    predictions.append("Error")
            except Exception as e:
                predictions.append(f"Error: {e}")

        # Show predictions after all are processed
        st.write("✅ All Predictions:")
        for i, pred in enumerate(predictions):
            st.write(f"Row {i + 1}: {pred}")
