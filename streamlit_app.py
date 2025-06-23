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

country = st.selectbox("Select Country", ["Indonesi]()
