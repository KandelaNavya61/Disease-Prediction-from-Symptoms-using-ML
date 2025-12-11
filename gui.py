import streamlit as st
import pandas as pd, os, joblib, numpy as np
from io import StringIO
from datetime import datetime


st.set_page_config(page_title="Disease Predictor", layout="wide")
hide_deploy_button = """
    <style>
    [data-testid="stAppDeployButton"] {display: none !important;}
    </style>
"""
st.markdown(hide_deploy_button, unsafe_allow_html=True)


st.title("Disease Prediction from Symptoms")

BASE = os.path.dirname(__file__)
DATA_CSV = os.path.join(BASE, "data", "health_disease_dataset.csv")
MODELS_DIR = os.path.join(BASE, "models")

# Load dataset and models
df = pd.read_csv(DATA_CSV)
feature_cols = [c for c in df.columns if c != "disease"]

if not os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")):
    st.warning("⚠️ Model not found! Run `python train.py` to train the model.")
else:
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

    # ------------------- PATIENT DETAILS SECTION -------------------
    st.sidebar.header("👤 Patient Information")

    patient_name = st.sidebar.text_input("Patient Name")
    patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, step=1)
    patient_gender = st.sidebar.selectbox("Gender", ["Select", "Female", "Male", "Other"])
    # patient_bloodGroup = st.sidebar.selectbox("Blood Group", ["Select", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    patient_bloodGroup = st.sidebar.text_input("BloodGroup")
    patient_contact = st.sidebar.text_input("Contact Number")
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # patient_email = st.sidebar.text_input("Email (optional)")
    # ---------------------------------------------------------------

    st.sidebar.header("🩹 Select Symptoms")
    selected = []
    cols = st.sidebar.columns(3)
    for i, symptom in enumerate(feature_cols):
        col = cols[i % 3]
        if col.checkbox(symptom):
            selected.append(symptom)

    if st.sidebar.button("🔍 Predict Disease"):
        # Validation
        if not patient_name or patient_gender == "Select" or patient_age == 0:
            st.warning("⚠️ Please fill in all required patient details before predicting.")
        elif not selected:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            # Convert selected symptoms into input vector
            x = np.zeros(len(feature_cols), dtype=int)
            for s in selected:
                if s in feature_cols:
                    x[feature_cols.index(s)] = 1

            pred_enc = model.predict(x.reshape(1, -1))[0]
            pred_label = le.inverse_transform([pred_enc])[0]

            # ----------------- Display Patient Info -----------------
            st.subheader("🧾 Patient Details")
            st.success(f"**Predicted Disease:** {pred_label}")
            st.markdown(f"**Name:** {patient_name}")
            st.markdown(f"**Age:** {patient_age}")
            st.markdown(f"**Gender:** {patient_gender}")
            if patient_bloodGroup:
                st.markdown(f"**Blood Group:** {patient_bloodGroup}")
            if patient_contact:
                st.markdown(f"**Contact:** {patient_contact}")
            # if patient_email:
            #     st.markdown(f"**Email:** {patient_email}")


            # ------------------ Cause dictionary ------------------
            disease_causes = {
                "Fungal Infection": "Caused by fungi that grow on skin, hair, or nails, especially in warm and moist areas.",
                "Malaria": "Caused by parasites spread through the bite of infected mosquitoes.",
                "Varicose Veins": "Caused when vein valves weaken, leading to blood pooling and twisted veins.",
                "Allergy": "Caused by the immune system overreacting to harmless things like pollen, dust, or food.",
                "Chickenpox": "Caused by the varicella-zoster virus, which spreads easily by coughing, sneezing, or touch.",
                "Hypothyroidism": "Thyroid gland makes too little hormone, often due to immune disease or medicines.",
                "GERD (Acid Reflux)": "Stomach acid flows back into the food pipe because the valve is weak.",
                "Dengue": "Spread by mosquito bites carrying dengue virus.",
                "Vertigo": "Problems in the inner ear or brain balance system.",
                "Chronic Cholestasis": "Bile flow from the liver is blocked, often due to liver disease or gallstones.",
                "Peptic Ulcer": "Open sores in the stomach or intestine due to bacteria (H. pylori) or painkillers.",
                "Acne": "Too much oil, clogged pores, bacteria, and inflammation.",
                "Drug Reaction": "Side effects or allergy from medicines.",
                "Hepatitis A": "Virus infection from food or water contaminated with stool.",
                "Urinary Tract Infection": "Bacteria (often E. coli) entering the urinary tract.",
                "Piles": "Swollen blood vessels from constipation, straining, or pregnancy.",
                "Hepatitis B": "Virus spread by contact with infected blood or body fluids.",
                "Psoriasis": "Immune system attacks skin cells, causing rapid growth.",
                "AIDS (Advanced HIV)": "HIV virus damages the immune system, spread by blood, sex, or needles.",
                "Hepatitis C": "Virus spread through infected blood (needles, transfusions)."
            }

            if pred_label in disease_causes:
                st.markdown(f"**Cause:** {disease_causes[pred_label]}")
            else:
                st.info("Cause information not available for this disease yet.")
            # ------------------------------------------------------

            # ------------------- SAVE PATIENT RECORD -------------------
            patient_data = {
                "Date": [date],
                "Name": [patient_name],
                "Age": [patient_age],
                "Gender": [patient_gender],
                "Blood Group": [patient_bloodGroup],
                "Contact": [patient_contact],
                # "Email": [patient_email],
                "Selected_Symptoms": [", ".join(selected)],
                "Predicted_Disease": [pred_label]
            }

            record_df = pd.DataFrame(patient_data)
            file_path = os.path.join(BASE, "patient_records.csv")

            # Append or create the CSV file
            if os.path.exists(file_path):
                existing = pd.read_csv(file_path)
                combined = pd.concat([existing, record_df], ignore_index=True)
                combined.to_csv(file_path, index=False)
            else:
                record_df.to_csv(file_path, index=False)

            st.success("✅ Patient record saved successfully!")
            # ------------------------------------------------------------

            # ------------------- DOWNLOAD BUTTON -------------------
            # csv_buffer = StringIO()
            # record_df.to_csv(csv_buffer, index=False)
            # st.download_button(
            #     label="⬇️ Download Patient Report (CSV)",
            #     data=csv_buffer.getvalue(),
            #     file_name=f"{patient_name}_report.csv",
            #     mime="text/csv"
            # )
            # --------------------------------------------------------




# import streamlit as st
# import pandas as pd, os, joblib, numpy as np
# from io import StringIO
# from datetime import datetime


# st.set_page_config(page_title="Disease Predictor", layout="wide")

# # Hide only deploy button (keep other menu icons visible)
# hide_deploy_button = """
#     <style>
#     [data-testid="stAppDeployButton"] {display: none !important;}
#     </style>
# """
# st.markdown(hide_deploy_button, unsafe_allow_html=True)


# st.title("Disease Prediction from Symptoms")

# BASE = os.path.dirname(__file__)
# DATA_CSV = os.path.join(BASE, "data", "health_disease_dataset.csv")
# MODELS_DIR = os.path.join(BASE, "models")

# # Load dataset and models
# df = pd.read_csv(DATA_CSV)
# feature_cols = [c for c in df.columns if c != "disease"]

# if not os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")):
#     st.warning("⚠️ Model not found! Run `python train.py` to train the model.")
# else:
#     le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
#     model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

#     # ------------------- PATIENT DETAILS SECTION -------------------
#     st.sidebar.header("👤 Patient Information")

#     patient_name = st.sidebar.text_input("Patient Name")
#     patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, step=1)
#     patient_gender = st.sidebar.selectbox("Gender", ["Select", "Female", "Male", "Other"])
#     patient_bloodGroup = st.sidebar.selectbox(
#         "Blood Group", ["Select", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
#     )
#     patient_weight = st.sidebar.number_input("Weight (kg)", min_value=0.0, step=0.1)
#     patient_bp = st.sidebar.text_input("Blood Pressure (e.g., 120/80 mmHg)")
#     patient_contact = st.sidebar.text_input("Contact Number")

#     date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     # ---------------------------------------------------------------

#     st.sidebar.header("🩹 Select Symptoms")
#     selected = []
#     cols = st.sidebar.columns(3)
#     for i, symptom in enumerate(feature_cols):
#         col = cols[i % 3]
#         if col.checkbox(symptom):
#             selected.append(symptom)

#     if st.sidebar.button("🔍 Predict Disease"):
#         # Validation
#         if not patient_name or patient_gender == "Select" or patient_age == 0 or patient_bloodGroup == "Select":
#             st.warning("⚠️ Please fill in all required patient details before predicting.")
#         elif not selected:
#             st.warning("⚠️ Please select at least one symptom.")
#         else:
#             # Convert selected symptoms into input vector
#             x = np.zeros(len(feature_cols), dtype=int)
#             for s in selected:
#                 if s in feature_cols:
#                     x[feature_cols.index(s)] = 1

#             pred_enc = model.predict(x.reshape(1, -1))[0]
#             pred_label = le.inverse_transform([pred_enc])[0]

#             # ----------------- Display Patient Info -----------------
#             st.subheader("🧾 Patient Details")
#             st.success(f"**Predicted Disease:** {pred_label}")
#             st.markdown(f"**Name:** {patient_name}")
#             st.markdown(f"**Age:** {patient_age}")
#             st.markdown(f"**Gender:** {patient_gender}")
#             st.markdown(f"**Blood Group:** {patient_bloodGroup}")
#             if patient_weight:
#                 st.markdown(f"**Weight:** {patient_weight} kg")
#             if patient_bp:
#                 st.markdown(f"**Blood Pressure:** {patient_bp}")
#             if patient_contact:
#                 st.markdown(f"**Contact:** {patient_contact}")
#             # ---------------------------------------------------------

#             # ------------------ Cause dictionary ------------------
#             disease_causes = {
#                 "Fungal Infection": "Caused by fungi that grow on skin, hair, or nails, especially in warm and moist areas.",
#                 "Malaria": "Caused by parasites spread through the bite of infected mosquitoes.",
#                 "Varicose Veins": "Caused when vein valves weaken, leading to blood pooling and twisted veins.",
#                 "Allergy": "Caused by the immune system overreacting to harmless things like pollen, dust, or food.",
#                 "Chickenpox": "Caused by the varicella-zoster virus, which spreads easily by coughing, sneezing, or touch.",
#                 "Hypothyroidism": "Thyroid gland makes too little hormone, often due to immune disease or medicines.",
#                 "GERD (Acid Reflux)": "Stomach acid flows back into the food pipe because the valve is weak.",
#                 "Dengue": "Spread by mosquito bites carrying dengue virus.",
#                 "Vertigo": "Problems in the inner ear or brain balance system.",
#                 "Chronic Cholestasis": "Bile flow from the liver is blocked, often due to liver disease or gallstones.",
#                 "Peptic Ulcer": "Open sores in the stomach or intestine due to bacteria (H. pylori) or painkillers.",
#                 "Acne": "Too much oil, clogged pores, bacteria, and inflammation.",
#                 "Drug Reaction": "Side effects or allergy from medicines.",
#                 "Hepatitis A": "Virus infection from food or water contaminated with stool.",
#                 "Urinary Tract Infection": "Bacteria (often E. coli) entering the urinary tract.",
#                 "Piles": "Swollen blood vessels from constipation, straining, or pregnancy.",
#                 "Hepatitis B": "Virus spread by contact with infected blood or body fluids.",
#                 "Psoriasis": "Immune system attacks skin cells, causing rapid growth.",
#                 "AIDS (Advanced HIV)": "HIV virus damages the immune system, spread by blood, sex, or needles.",
#                 "Hepatitis C": "Virus spread through infected blood (needles, transfusions)."
#             }

#             if pred_label in disease_causes:
#                 st.markdown(f"**Cause:** {disease_causes[pred_label]}")
#             else:
#                 st.info("Cause information not available for this disease yet.")
#             # ------------------------------------------------------

#             # ------------------- SAVE PATIENT RECORD -------------------
#             patient_data = {
#                 "Date": [date],
#                 "Name": [patient_name],
#                 "Age": [patient_age],
#                 "Gender": [patient_gender],
#                 "Blood Group": [patient_bloodGroup],
#                 "Weight (kg)": [patient_weight],
#                 "Blood Pressure": [patient_bp],
#                 "Contact": [patient_contact],
#                 "Selected_Symptoms": [", ".join(selected)],
#                 "Predicted_Disease": [pred_label]
#             }

#             record_df = pd.DataFrame(patient_data)
#             file_path = os.path.join(BASE, "patient_records.csv")

#             # Append or create the CSV file
#             if os.path.exists(file_path):
#                 existing = pd.read_csv(file_path)
#                 combined = pd.concat([existing, record_df], ignore_index=True)
#                 combined.to_csv(file_path, index=False)
#             else:
#                 record_df.to_csv(file_path, index=False)

#             st.success("✅ Patient record saved successfully!")
#             # ------------------------------------------------------------
