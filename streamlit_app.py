import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np

class WaterNet(nn.Module):
    def __init__(self):
        super(WaterNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)


model = WaterModel()
model.load_state_dict(torch.load("water_model.pt", map_location=torch.device("cpu")))
model.eval()

scaler = joblib.load("scaler.pkl")

st.title(":droplet: Water Prediction App")
st.markdown("Predict **Population using Safely Managed Drinking Water (%)** based on input data.")

country = st.selectbox("Select Country", ["Indonesia", "Thailand", "Malaysia", "Philippines", "Vietnam", "Laos", "Cambodia", "Singapore", "Myanmar", "Brunei"])
area = st.selectbox("Area Type", ["Urban", "Rural", "Overall"])
year = st.number_input("Year", min_value=2000, max_value=2030, value=2025)
population = st.number_input("Total Population", min_value=1000, step=1000)
stress = st.number_input("Estimated Water Stress (%)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

if st.button("Predict Safe Drinking Water %"):
    input_data = np.array([[year, population, stress]])
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    st.success(f":crystal_ball: Predicted Safely Managed Drinking Water Access: **{prediction * 100:.2f}%**")

st.markdown("---")
st.info("This model predicts access to safe drinking water. Extend it to also predict water stress using a separate model.")
