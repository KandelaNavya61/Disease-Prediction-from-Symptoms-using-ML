import streamlit as st
import pandas as pd, os, joblib, numpy as np

st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("Disease Prediction from Symptoms")
BASE = os.path.dirname(__file__)
DATA_CSV = os.path.join(BASE, "data", "health_disease_dataset.csv")
MODELS_DIR = os.path.join(BASE, "models")

# Load data and models
df = pd.read_csv(DATA_CSV)
feature_cols = [c for c in df.columns if c != "disease"]

if not os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")):
    st.warning("No trained model found. Please run `python train.py` first to train and save models.")
else:
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

    st.sidebar.header("Select Symptoms")
    selected = []
    cols = st.sidebar.columns(3)
    for i, symptom in enumerate(feature_cols):
        col = cols[i % 3]
        if col.checkbox(symptom):
            selected.append(symptom)

    if st.sidebar.button("Predict"):
        x = np.zeros(len(feature_cols), dtype=int)
        for s in selected:
            if s in feature_cols:
                x[feature_cols.index(s)] = 1
        pred_enc = model.predict(x.reshape(1, -1))[0]
        pred_label = le.inverse_transform([pred_enc])[0]
        st.success(f"Predicted disease: {pred_label}")

         # ------------------ Cause dictionary ------------------
        disease_causes = {
            "Fungal Infection": "Caused by fungi that grow on skin, hair, or nails, especially in warm and moist areas.",
            "Malaria": "Caused by parasites spread through the bite of infected mosquitoes.",
            "Varicose Veins": "Caused when vein valves weaken, leading to blood pooling and twisted veins.",
            "Allergy": "Caused by the immune system overreacting to harmless things like pollen, dust, or food.",
            "Chickenpox": "Caused by the varicella-zoster virus, which spreads easily by coughing, sneezing, or touch.",
            # 👉 Add more diseases here
        }

        if pred_label in disease_causes:
            st.markdown(f"**Cause:** {disease_causes[pred_label]}")
        else:
            st.info("Cause information not available for this disease yet.")
        # ------------------------------------------------------

    st.sidebar.markdown("---")
    st.sidebar.markdown("If no model is present, run `python train.py` in this folder to train models.")

st.markdown("---")
st.header("Dataset sample")
st.dataframe(df.sample(5))

