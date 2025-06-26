import streamlit as st
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_water = load_lottieurl("https://lottie.host/a28fc662-6787-4c09-b221-b384aa96f335/dmINnOLGwC.json")

st.title(":droplet: Welcome to Water Prediction App!")
st_lottie(lottie_water, height=300, key="water")

st.markdown("""
### 🌍 Predict access to safely managed drinking water from South East Asian countries!

Use this tool to explore:
- **Predictions** based on population, year, water stress, and more
- **How the model works**
- **Data insights** from Southeast Asia based on UNICEF's data

Go to the **sidebar** and choose a page to get started!
""")
