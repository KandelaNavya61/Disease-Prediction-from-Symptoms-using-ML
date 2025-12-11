import os, joblib, pandas as pd, numpy as np

BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")
DATA_CSV = os.path.join(BASE, "data", "health_disease_dataset.csv")

le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

df = pd.read_csv(DATA_CSV)
feature_cols = [c for c in df.columns if c != "disease"]

def predict_from_symptom_names(symptom_names):
    x = [0]*len(feature_cols)
    for s in symptom_names:
        if s in feature_cols:
            idx = feature_cols.index(s)
            x[idx] = 1
    arr = np.array(x).reshape(1, -1)
    pred_enc = model.predict(arr)[0]
    return le.inverse_transform([pred_enc])[0]

if __name__ == "__main__":
    print("Enter comma-separated symptom names (exact as columns).")
    s = input("Symptoms: ")
    names = [x.strip() for x in s.split(",") if x.strip()]
    print("Predicted disease:", predict_from_symptom_names(names))