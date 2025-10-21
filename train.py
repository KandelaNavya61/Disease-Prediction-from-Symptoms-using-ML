import pandas as pd
import os, joblib, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

BASE = os.path.dirname(__file__)
DATA_CSV = os.path.join(BASE, "data", "health_disease_dataset.csv")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_CSV)
print("Dataset shape:", df.shape)

X = df.drop(columns=["disease"])
y = df["disease"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
print("Classes:", list(le.classes_))

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

models = {
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "naive_bayes": GaussianNB()
}

results = {}
for name, model in models.items():
    print("\nTraining", name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    results[name] = (acc, model)
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))

best_name = max(results.keys(), key=lambda k: results[k][0])
best_acc, best_model = results[best_name]
print(f"Best model: {best_name} (accuracy={best_acc:.4f})")
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
print("Saved models to", MODELS_DIR)