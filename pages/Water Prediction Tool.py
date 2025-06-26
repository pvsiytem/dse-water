import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np

st.set_page_config(page_title="Water Access Predictor", page_icon="💧", layout="centered")

st.markdown("<h1 style='text-align: center; color: #0066cc;'>🔮 Water Prediction Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Estimate the <strong>percentage of population with access to safe drinking water</strong> across Southeast Asia.</p>", unsafe_allow_html=True)
st.markdown("---")

class WaterNet(nn.Module):
    def __init__(self, input_size):
        super(WaterNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
num_features = scaler.mean_.shape[0] + encoder.get_feature_names_out().shape[0]

model = WaterNet(input_size=num_features)
model.load_state_dict(torch.load("water_model.pt", map_location=torch.device("cpu")))
model.eval()

st.subheader("📥 Enter Parameters")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    country = col1.selectbox("🌐 Country", [
        "Indonesia", "Thailand", "Malaysia", "Philippines", "Vietnam", 
        "Laos", "Cambodia", "Singapore", "Myanmar", "Brunei"
    ])

    area = col2.selectbox("🏘️ Area Type", ["Urban", "Rural", "Overall"])

    year = st.slider("📅 Year", 2000, 2050, 2025)
    population = st.number_input("👥 Total Population", min_value=1000, step=10000)
    stress = st.slider("💧 Estimated Water Stress (%)", 0.0, 1.0, 0.3, 0.01)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    with st.spinner("Running model..."):
        numeric_input = scaler.transform([[year, population, stress]])
        categorical_input = encoder.transform([[country, area]])
        full_input = np.hstack([numeric_input, categorical_input])
        input_tensor = torch.tensor(full_input, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(input_tensor).item()
        
        percent = prediction * 100

    st.success("✅ Prediction Complete!")
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background-color: #e6f2ff; border-radius: 10px;'>
        <h2 style='color: #0099ff;'>🌟 Estimated Access to Safe Drinking Water</h2>
        <p style='font-size: 32px; font-weight: bold;'>{percent:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

st.caption("Backend: PyTorch + Streamlit")
