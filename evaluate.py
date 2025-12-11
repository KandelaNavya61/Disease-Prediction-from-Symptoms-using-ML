import pandas as pd, os, joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE = os.path.dirname(__file__)
DATA_CSV = os.path.join(BASE, "data", "health_disease_dataset.csv")
MODELS_DIR = os.path.join(BASE, "models")

df = pd.read_csv(DATA_CSV)
X = df.drop(columns=["disease"])
y = df["disease"]

le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
y_enc = le.transform(y)

for fname in ["decision_tree.pkl","random_forest.pkl","naive_bayes.pkl","best_model.pkl"]:
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        print("Model not found:", fname)
        continue
    model = joblib.load(path)
    y_pred = model.predict(X)
    acc = accuracy_score(y_enc, y_pred)
    print(f"Model: {fname} - Accuracy: {acc:.4f}")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_enc, y_pred))
    print("-"*50)