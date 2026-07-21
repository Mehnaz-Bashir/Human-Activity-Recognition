import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("../models/best_har_model.pkl")
feature_names = joblib.load("../models/feature_names.pkl")


activity_map = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏃 HAR Dashboard")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.metric(
    label="Model Accuracy",
    value="98.59%"
)

st.sidebar.markdown("---")

st.sidebar.write("### Activity Classes")

for value in activity_map.values():
    st.sidebar.write("✅", value)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Developer:
Mehnaz Bashir

MCA

Lovely Professional University
"""
)

# -----------------------------
# Main Title
# -----------------------------
st.title("🏃 Human Activity Recognition")

st.write(
"""
Predict Human Activities using Smartphone Sensor Data.
"""
)

st.markdown("---")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Sensor CSV",
    type="csv"
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("File Uploaded Successfully!")

    st.subheader("Dataset Preview")

    st.dataframe(data.head())

    st.write("Rows :", data.shape[0])
    st.write("Columns :", data.shape[1])

    # -----------------------------
    # Predict Button
    # -----------------------------

    if st.button("🚀 Predict Activities"):

        data = data[feature_names]

        predictions = model.predict(data)

        predicted_labels = [
            activity_map[p]
            for p in predictions
        ]

        data["Predicted Activity"] = predicted_labels

        st.success("Prediction Completed!")

        # -----------------------------
        # Prediction Results
        # -----------------------------

        st.subheader("Prediction Results")

        st.dataframe(data)

        # -----------------------------
        # Summary
        # -----------------------------

        st.subheader("Prediction Summary")

        summary = data["Predicted Activity"].value_counts()

        st.dataframe(summary)

        # -----------------------------
        # Charts
        # -----------------------------

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots(figsize=(6,4))

            summary.plot(
                kind="bar",
                ax=ax
            )

            ax.set_title("Activity Distribution")

            ax.set_ylabel("Count")

            st.pyplot(fig)

        with col2:

            fig2, ax2 = plt.subplots(figsize=(6,6))

            summary.plot(
                kind="pie",
                autopct="%1.1f%%",
                ax=ax2
            )

            ax2.set_ylabel("")

            ax2.set_title("Activity Percentage")

            st.pyplot(fig2)

        # -----------------------------
        # Download
        # -----------------------------

        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
