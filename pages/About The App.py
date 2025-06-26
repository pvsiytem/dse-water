import streamlit as st

st.title("About The App")
st.markdown("""

### 💧 Why it's made:
- Made as a requirement of the ASEAN DSE competition, using SDG 6 as its main theme and problem to solve
- By making this app, users can use the safely drinkable water prediction app to make their own trajectory of a country's area's outlook whether it's positive or negative.

### 🤖 How it works:
- The app is centered around a custom **neural network** trained on water access and demographic data based on ASEAN safely managed drinking water.
- Built with **PyTorch**
- Uses **StandardScaler** for numeric features and **OneHotEncoder** for categories
- It is relatively reliable due to the 99% accuracy the model had

### 🏗️ Model Architecture:
- Input Layer → Linear(64) → ReLU
- Hidden Layers → 32, 16, 8 neurons with ReLU
- Output Layer → Sigmoid activation (predicts % access, making sure it doesn't exceeds 100%)

### 📁 Input Features:
- **Year**
- **Population**
- **Water Stress (%)**
- **Country**
- **Area Type**

### Data Reference:
- Adding It Soon!!
""")
