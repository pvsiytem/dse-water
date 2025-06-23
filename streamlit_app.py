import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np

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
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

num_features = scaler.mean_.shape[0] + encoder.get_feature_names_out().shape[0]

model = WaterNet(input_size=num_features)
model.load_state_dict(torch.load("water_model.pt", map_location=torch.device("cpu")))
model.eval()

st.title(":droplet: Water Prediction App")
st.markdown("Predict **Population using Safely Managed Drinking Water Service (%)** based on input data.")

country = st.selectbox("Select Country", ["Indonesia", "Thailand", "Malaysia", "Philippines", "Vietnam", "Laos", "Cambodia", "Singapore", "Myanmar", "Brunei"])
area = st.selectbox("Area Type", ["Urban", "Rural", "Overall"])
year = st.number_input("Year", min_value=2000, max_value=2050, value=2025)
population = st.number_input("Total Population", min_value=1000, step=1000)
stress = st.number_input("Estimated Water Stress (%)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

if st.button("Predict Safe Drinking Water %"):
    numeric_input = scaler.transform([[year, population, stress]])
    categorical_input = encoder.transform([[country, area]])
    full_input = np.hstack([numeric_input, categorical_input])
    input_tensor = torch.tensor(full_input, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    st.success(f":crystal_ball: Predicted Safely Managed Drinking Water Access: **{prediction * 100:.2f}%**")

st.markdown("---")
st.info("Now considers both country and area in predictions. You can expand it further by adding more predictors or visualization!")
