import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS (yeh sab sirf look & feel ke liye hai)
# =========================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fb;
        border: 1px solid #eaeaea;
        border-radius: 12px;
        padding: 10px 15px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        opacity: 0.9;
        color: white;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# ROBUST MODEL LOADING
# Deployment ke time folder structure alag ho sakta hai
# (e.g. models/ app ke saath hi ho, ya ek level upar).
# Isliye multiple possible locations check karte hain.
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

POSSIBLE_MODEL_DIRS = [
    os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "..", "models"),
    os.path.join(BASE_DIR, "model"),
    BASE_DIR,
]

@st.cache_resource(show_spinner=False)
def load_model_and_features():
    for folder in POSSIBLE_MODEL_DIRS:
        model_path = os.path.join(folder, "best_har_model.pkl")
        feature_path = os.path.join(folder, "feature_names.pkl")
        if os.path.exists(model_path) and os.path.exists(feature_path):
            model = joblib.load(model_path)
            feature_names = joblib.load(feature_path)
            return model, feature_names, folder
    return None, None, None

with st.spinner("Loading model..."):
    model, feature_names, found_dir = load_model_and_features()

if model is None:
    st.error(
        "❌ Model files nahi mile (best_har_model.pkl / feature_names.pkl).\n\n"
        "Checked these folders:\n" + "\n".join(f"- {p}" for p in POSSIBLE_MODEL_DIRS) +
        "\n\nMake sure `models/` folder repo mein app.py ke sahi relative path par committed ho "
        "(Git LFS use kiya hai to Streamlit Cloud LFS files auto-pull nahi karta — normal upload use karo)."
    )
    st.stop()

activity_map = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

ACTIVITY_ICONS = {
    "WALKING": "🚶",
    "WALKING_UPSTAIRS": "⬆️",
    "WALKING_DOWNSTAIRS": "⬇️",
    "SITTING": "🪑",
    "STANDING": "🧍",
    "LAYING": "🛌",
}

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("🏃 HAR Dashboard")
    st.success("✅ Model Loaded")
    st.caption(f"Loaded from: `{os.path.relpath(found_dir, BASE_DIR) or '.'}`")

    st.metric(label="Model Accuracy", value="98.59%")

    st.markdown("---")
    st.write("### Activity Classes")
    for value in activity_map.values():
        st.write(f"{ACTIVITY_ICONS.get(value, '✅')} {value}")

    st.markdown("---")
    st.info(
        "**Developer:** Mehnaz Bashir\n\n"
        "**Program:** MCA\n\n"
        "**Institute:** Lovely Professional University"
    )

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="main-header">
    <h1>🏃 Human Activity Recognition</h1>
    <p>Predict human activities (walking, sitting, standing, laying, etc.) from smartphone accelerometer & gyroscope sensor data.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# FILE UPLOAD
# =========================================================
st.subheader("📂 Upload Sensor Data")
uploaded_file = st.file_uploader(
    "Upload a CSV file with the same feature columns used during training",
    type="csv"
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"⚠️ Could not read the CSV file: {e}")
        st.stop()

    st.success("File uploaded successfully!")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", data.shape[0])
    c2.metric("Columns", data.shape[1])
    c3.metric("Required Features", len(feature_names))

    with st.expander("🔍 Preview Dataset", expanded=True):
        st.dataframe(data.head(), use_container_width=True)

    # ---------------------------------------------------
    # Validate that required columns exist
    # ---------------------------------------------------
    missing_cols = [f for f in feature_names if f not in data.columns]

    if missing_cols:
        st.error(
            f"⚠️ Uploaded file is missing {len(missing_cols)} required feature column(s). "
            "Prediction can't run until the CSV matches the model's expected columns."
        )
        with st.expander("See missing columns"):
            st.write(missing_cols)
        st.stop()

    st.markdown("---")

    if st.button("🚀 Predict Activities", use_container_width=True):
        with st.spinner("Running predictions..."):
            model_input = data[feature_names]
            predictions = model.predict(model_input)
            predicted_labels = [activity_map.get(p, str(p)) for p in predictions]

            result = data.copy()
            result["Predicted Activity"] = predicted_labels

        st.success("✅ Prediction Completed!")

        st.subheader("📊 Prediction Results")
        st.dataframe(result, use_container_width=True)

        # -----------------------------------------------
        # Summary
        # -----------------------------------------------
        st.subheader("📈 Prediction Summary")

        summary = result["Predicted Activity"].value_counts()

        summary_cols = st.columns(len(summary))
        for col, (activity, count) in zip(summary_cols, summary.items()):
            col.metric(
                label=f"{ACTIVITY_ICONS.get(activity, '')} {activity}",
                value=int(count)
            )

        # -----------------------------------------------
        # Charts
        # -----------------------------------------------
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            summary.plot(kind="bar", ax=ax, color="#2575fc")
            ax.set_title("Activity Distribution")
            ax.set_ylabel("Count")
            ax.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)

        with chart_col2:
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            colors = plt.cm.cool_r(range(0, 256, max(1, 256 // len(summary))))
            summary.plot(kind="pie", autopct="%1.1f%%", ax=ax2, colors=colors)
            ax2.set_ylabel("")
            ax2.set_title("Activity Percentage")
            st.pyplot(fig2)

        # -----------------------------------------------
        # Download
        # -----------------------------------------------
        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    st.info("👆 Upload a CSV file to get started.")
