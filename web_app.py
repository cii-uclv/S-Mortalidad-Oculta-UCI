import pandas as pd
import pickle
import cloudpickle
import streamlit as st
import lime.lime_tabular
import shap
import google.generativeai as genai

# import captum
from interpret.blackbox import LimeTabular
from interpret import show
from interpret.blackbox import ShapKernel
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from joblib import load
import os
import dotenv
import openai

set_visualize_provider(InlineProvider())
from interpret import show
import torch
import torch.nn as nn

# from captum.attr import IntegratedGradients
# from captum.attr import Saliency
import numpy as np
import matplotlib.pyplot as plt


# Define provided data
data = """
0= Empty
1= Exogenous intoxication
2= Coma
3= Severe traumatic brain injury
4= Post-thoracotomy
5= Post-laparotomy
6= Post-amputation
7= Post-neurology surgery
8= Recovered cardiac arrest
9= Metabolic encephalopathy
10= Hypoxic encephalopathy
11= Incomplete hanging
12= Decompensated heart failure
13= Severe obstetric condition
14= Decompensated COPD
15= ARDS
16= BNB-EH
17= BNB-IH
18= BNV
19= Myocarditis
20= Leptospirosis
21= Severe sepsis
22= DMO
23= Septic shock
24= Hypovolemic shock
25= Cardiogenic shock
26= Myocardial infarction
27= Polytrauma
28= Myasthenic crisis
29= Hypertensive emergency
30= Status asthmaticus
31= Status epilepticus
32= Pancreatitis 
33= Fat embolism
34= Stroke
35= Sleep apnea syndrome
36= Digestive bleeding
37= Chronic renal failure
38= Acute renal failure
39= Renal transplant
40= Guillain-Barré
41= AV block
42= Obstetric embolism
43= Aspiration pneumonia
44= Neuroleptic malignant syndrome
45= Diabetic ketoacidosis
46= Meningitis
47= Pulmonary edema
48= Others
"""

# Process data and create dictionaries
num_to_desc = {}
desc_to_num = {}

lines = [line.strip() for line in data.strip().split("\n")]

for line in lines:
    key_part, value = line.split("=", 1)  # Split at first '='
    key = int(key_part.strip())
    value = value.strip()  # Remove surrounding whitespace
    num_to_desc[key] = value
    desc_to_num[value] = key

st.set_page_config(layout="wide")

# Load fixed model
path = r"./Models/"
model = load("new_workflow.joblib")

# Load explanations
with open("Explainers/ig_explainer.pkl", "rb") as archivo:
    ig_exp = pickle.load(archivo)


# Function to get user input
def get_user_input():
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=20, step=1)
    diag_ing1 = desc_to_num[
        st.sidebar.selectbox(
            label="Admission Diagnosis 1", options=list(desc_to_num.keys())
        )
    ]
    diag_ing2 = desc_to_num[
        st.sidebar.selectbox(
            label="Admission Diagnosis 2", options=list(desc_to_num.keys())
        )
    ]
    diag_egr2 = desc_to_num[
        st.sidebar.selectbox(
            label="Discharge Diagnosis 2", options=list(desc_to_num.keys())
        )
    ]
    apache = st.sidebar.number_input(
        "APACHE II", min_value=0, max_value=40, value=18, step=1
    )
    tiempo_vam = st.sidebar.number_input(
        "Ventilator Time", min_value=1, max_value=200, value=5, step=1
    )

    user_data = {
        "Edad": age,
        "Diag.Ing1": diag_ing1,
        "Diag.Ing2": diag_ing2,
        "Diag.Egr2": diag_egr2,
        "APACHE": apache,
        "TiempoVAM": tiempo_vam,
    }
    features = pd.DataFrame(user_data, index=[0])
    return features


feature_names = [
    "Age",
    "Adm.Diag1",
    "Adm.Diag2",
    "Dis.Diag2",
    "APACHE",
    "VentilatorTime",
]


def plot_feature_importances(feature_names, importances):
    """
    Plot feature importance with differentiated colors
    for positive (orange) and negative (blue) contributions.
    """
    importances = np.array(importances).flatten()

    attrib_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

    attrib_df["AbsImportance"] = attrib_df["Importance"].abs()
    attrib_df = attrib_df.sort_values(by="AbsImportance", ascending=False)

    colors = attrib_df["Importance"].apply(lambda x: "orange" if x > 0 else "blue")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(attrib_df["Feature"], attrib_df["Importance"], color=colors)
    ax.set_xlabel("Relevance")
    ax.set_title("Feature Relevance (Colors: Orange=Positive, Blue=Negative)")
    ax.axvline(0, color="red", linestyle="--")
    ax.invert_yaxis()

    return fig


# Streamlit state variables
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.input_df = None

# Get user input
input_df = get_user_input()
input_df_original = input_df.copy()
input_df_original.rename(
    columns={
        "Edad": "Age",
        "Diag.Ing1": "Adm.Diag1",
        "Diag.Ing2": "Adm.Diag2",
        "Diag.Egr2": "Dis.Diag2",
        "APACHE": "APACHE",
        "TiempoVAM": "VentilatorTime",
    },
    inplace=True,
)


# Convert DataFrame to PyTorch tensor
input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
input_tensor = input_tensor.unsqueeze(0)

st.title("Prediction of Non-Survival at ICU Discharge")
st.write("This tool supports prediction of patient non-survival at ICU discharge.")

st.write("#### Patient Characteristics")
st.write(input_df_original)

# Prediction button
predict = st.sidebar.button("Predict")

# Explanation button
explain = st.sidebar.button("Explain")

if predict:
    # Calculate probability prediction
    prob = model.predict_proba(input_df)[:, 1][0]
    st.session_state.prediction = prob

if st.session_state.prediction is not None:
    st.write("### Probability of Non-Survival (Cut 50%)")
    st.write(f"##### {st.session_state.prediction:.2%}")

if explain and (st.session_state.prediction is None):
    st.warning("First, make a prediction.")
elif explain:
    st.write(f"### Model Explanation")

    # 1. Calculate Feature Importances
    attr = ig_exp.attribute(input_tensor, target=0)
    attributions_np = attr.numpy()

    # 2. Visualize the importance
    fig = plot_feature_importances(feature_names, attributions_np)
    st.pyplot(fig, use_container_width=True)

    # --- Gemini API Configuration ---
    # Ensure you have GEMINI_API_KEY in your .env file
    apikey = dotenv.get_key(dotenv_path=".env", key_to_get="GEMINI_API_KEY")
    genai.configure(api_key=apikey)

    # --- Prompt Construction (ICU & Hidden Mortality Context) ---
    base_prompt = """You are a specialist in Intensive Care Medicine and clinical data science. 
    Your goal is to help clinicians understand why an AI model has flagged a patient for 'hidden mortality' risk—a situation where 
    the patient's clinical deterioration might not be immediately obvious from bedside observation alone."""

    definitions = """
    Strictly follow this format:
    Context: Predicting hidden mortality in the Intensive Care Unit (ICU).
    Explanation: Analysis of the clinical variables driving the risk assessment.
    Input format: (Clinical Parameter, Observed Value, Feature Importance Score).
    Narrative: A professional, human-readable clinical synthesis of the explanation, focusing on the most impactful features.
    """

    # Flatten tensors for processing
    x = input_tensor.numpy().flatten()
    attributions_np = attributions_np.flatten()
    explanations_data = ""

    # Prepare the data string for the prompt
    for i in range(len(feature_names)):
        if (
            feature_names[i] == "Adm.Diag1"
            or feature_names[i] == "Adm.Diag2"
            or feature_names[i] == "Dis.Diag2"
        ):
            patient_value_desc = num_to_desc[int(x[i])]
            explanations_data += (
                f"({feature_names[i]}, {patient_value_desc}, {attributions_np[i]})"
            )
        else:
            explanations_data += f"({feature_names[i]}, {x[i]}, {attributions_np[i]})"
        if i < len(feature_names) - 1:
            explanations_data += ", "

    full_context = f"""
    Input Data: {explanations_data}
    Input format: (Feature Name, Patient Value, Feature Importance)
    Context: The model identifies hidden mortality patterns in critically ill patients.
    """

    output_instruction = """
    Provide ONLY the 'Narrative' field. Do so immediately, without any introductory remarks or conversational filler. 
    Begin your response exactly with the word 'Narrative:'.
    """

    # --- Gemini API Call ---
    # 'gemini-1.5-flash' is fast and cost-effective; use 'gemini-1.5-pro' for more complex medical reasoning.
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    prompt_final = base_prompt + definitions + full_context + output_instruction

    response = model.generate_content(
        prompt_final,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,  # Balanced for professional yet descriptive output
        ),
    )

    # --- Displaying the Output ---
    # Extracting text and removing the "Narrative:" prefix for a cleaner UI
    narrative_output = response.text.replace("Narrative:", "").strip()
    st.info(narrative_output)
