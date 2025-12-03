import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import gdown

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Health Risk AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Load Models from Google Drive ---
# --- Função para Baixar Arquivos ---
@st.cache_resource
def download_model_if_not_exists(file_id, output_name):
    """
    Baixa o arquivo do Google Drive se ele não existir localmente.
    """
    if not os.path.exists(output_name):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_name, quiet=False)
    return output_name

# --- Carregar Modelos ---
@st.cache_resource
def load_models():
    # -----------------------------------------------------------
    # ⚠️ IDs DOS ARQUIVOS DO DRIVE
    # -----------------------------------------------------------
    id_scaler = "1Jyuc_k1VG_i3_U-vGDZc-mJ8EX_nsJj3"
    id_alcool = "1uqRaMs5Yk3BINYntLzGX85vAPZ95Lyyb"
    id_fumo = "1UX8xszo5CQdv-75Kgl3aGkZTDoCHPfni"
    # -----------------------------------------------------------

    try:
        with st.spinner('Downloading models from cloud (this may take a while on first run)...'):
            # Baixa os arquivos se não existirem
            download_model_if_not_exists(id_scaler, "model_data/scaler.pkl")
            download_model_if_not_exists(id_alcool, "model_data/modelo_alcool.pkl")
            download_model_if_not_exists(id_fumo, "model_data/modelo_fumo.pkl")

            # Carrega os modelos
            scaler = joblib.load("model_data/scaler.pkl")
            modelo_alcool = joblib.load("model_data/modelo_alcool.pkl")
            modelo_fumo = joblib.load("model_data/modelo_fumo.pkl")
        
        return scaler, modelo_alcool, modelo_fumo
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

scaler, modelo_alcool, modelo_fumo = load_models()

# --- 4. Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("Health AI")
    st.info(
        "This application uses Machine Learning to predict health behaviors based on clinical data.\n\n"
        "**Models Used:** Random Forest\n"
        "**Accuracy:** ~99% (Synthetic Test)"
    )
    st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only and does not replace professional medical advice.")

# --- 5. Main Content ---
if not scaler:
    st.error("🚨 **Error:** Model files (.pkl) not found. Please ensure `scaler.pkl`, `modelo_alcool.pkl`, and `modelo_fumo.pkl` are in the same directory.")
    st.stop()

st.title("🩺 Patient Health Profile Predictor")
st.markdown("### Enter the patient's clinical data below:")

with st.form("health_form"):
    
    # --- Section 1: Demographics & Body ---
    st.subheader("👤 Demographics & Body Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sex_input = st.selectbox("Sex", ["Male", "Female"])
        sex = 1 if sex_input == "Male" else 0
    with c2:
        age = st.number_input("Age", 18, 100, 30)
    with c3:
        height = st.number_input("Height (cm)", 100, 220, 175)
    with c4:
        weight = st.number_input("Weight (kg)", 30, 200, 75)
    
    waistline = st.slider("Waist Circumference (cm)", 40.0, 150.0, 80.0)

    # --- Section 2: Senses & Vitals ---
    with st.expander("👁️ Senses & Vital Signs", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1: sight_left = st.number_input("Sight (Left)", 0.1, 2.5, 1.0, step=0.1)
        with col2: sight_right = st.number_input("Sight (Right)", 0.1, 2.5, 1.0, step=0.1)
        with col3: hear_left = st.selectbox("Hearing (Left)", [1.0, 2.0], format_func=lambda x: "Normal" if x==1 else "Abnormal")
        with col4: hear_right = st.selectbox("Hearing (Right)", [1.0, 2.0], format_func=lambda x: "Normal" if x==1 else "Abnormal")
        
        st.markdown("---")
        col5, col6, col7 = st.columns(3)
        with col5: sbp = st.number_input("Systolic BP (mmHg)", 70, 220, 120)
        with col6: dbp = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)
        with col7: blds = st.number_input("Fasting Glucose (mg/dL)", 50, 400, 95)

    # --- Section 3: Lab Results (Lipids & Urine) ---
    with st.expander("🩸 Lipid Panel & Urinalysis"):
        c1, c2 = st.columns(2)
        with c1:
            tot_chole = st.number_input("Total Cholesterol (mg/dL)", 50, 500, 190)
            hdl = st.number_input("HDL Cholesterol (mg/dL)", 10, 200, 55)
            ldl = st.number_input("LDL Cholesterol (mg/dL)", 10, 400, 110)
        with c2:
            triglyceride = st.number_input("Triglycerides (mg/dL)", 20, 1000, 130)
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 14.5, step=0.1)
            urine_protein = st.selectbox("Urine Protein", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], help="1: Negative, 6: Highly Positive")

    # --- Section 4: Liver & Kidney Function ---
    with st.expander("🧪 Liver & Kidney Function"):
        c1, c2, c3, c4 = st.columns(4)
        with c1: serum_creatinine = st.number_input("Serum Creatinine", 0.1, 20.0, 0.9, step=0.1)
        with c2: sgot_ast = st.number_input("AST (SGOT)", 5, 1000, 25)
        with c3: sgot_alt = st.number_input("ALT (SGPT)", 5, 1000, 25)
        with c4: gamma_gtp = st.number_input("Gamma GTP", 5, 1000, 30, help="High values may indicate alcohol consumption")

    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button("🚀 Run Prediction")

# --- 6. Prediction Logic ---
if submit_btn:
    # Prepare Data (Order must match training)
    features = [
        sex, age, height, weight, waistline, sight_left, sight_right,
        hear_left, hear_right, sbp, dbp, blds, tot_chole, hdl, ldl,
        triglyceride, hemoglobin, urine_protein, serum_creatinine,
        sgot_ast, sgot_alt, gamma_gtp
    ]
    
    # Create DataFrame
    cols = ['sex', 'age', 'height', 'weight', 'waistline', 'sight_left', 'sight_right', 
            'hear_left', 'hear_right', 'SBP', 'DBP', 'BLDS', 'tot_chole', 'HDL_chole', 
            'LDL_chole', 'triglyceride', 'hemoglobin', 'urine_protein', 'serum_creatinine', 
            'SGOT_AST', 'SGOT_ALT', 'gamma_GTP']
    
    input_df = pd.DataFrame([features], columns=cols)

    # Scale Data
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Error scaling data: {e}. Ensure feature count matches the model.")
        st.stop()

    # Predict
    pred_alcohol = modelo_alcool.predict(input_scaled)[0]
    pred_smoke = modelo_fumo.predict(input_scaled)[0]

    # --- 7. Display Results ---
    st.markdown("---")
    st.header("📋 Prediction Results")

    col1, col2 = st.columns(2)

    # Alcohol Result
    with col1:
        st.subheader("Alcohol Status")
        if pred_alcohol == 1:
            st.error("🍺 **Drinker Detected**")
            st.caption("The model identified patterns consistent with alcohol consumption.")
        else:
            st.success("💧 **Non-Drinker**")
            st.caption("No significant patterns of alcohol consumption detected.")

    # Smoke Result
    with col2:
        st.subheader("Smoking Status")
        if pred_smoke == 1.0:
            st.success("🌿 **Non-Smoker**")
            st.caption("Healthy lifestyle regarding tobacco.")
        elif pred_smoke == 2.0:
            st.warning("🚬 **Ex-Smoker**")
            st.caption("Patterns indicate a history of smoking.")
        elif pred_smoke == 3.0:
            st.error("🔥 **Active Smoker**")
            st.caption("Strong indicators of active tobacco use.")