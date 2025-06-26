import streamlit as st
from streamlit_lottie import st_lottie
from streamlit.components.v1 import html
import requests

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.set_page_config(page_title="Water Prediction App", layout="wide")

lottie_url = "https://lottie.host/a28fc662-6787-4c09-b221-b384aa96f335/dmINnOLGwC.json"

background_lottie_html = f"""
<div style="position: fixed; width: 100%; height: 100%; z-index: -1; top: 0; left: 0;">
  <lottie-player
    src="{lottie_url}"
    background="transparent"
    speed="1"
    loop
    autoplay
    style="width: 100%; height: 100%;">
  </lottie-player>
</div>

<script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
"""

html(background_lottie_html, height=0, width=0)

st.title(":droplet: Welcome to Water Prediction App!")

st.markdown("""
### 🌍 Predict access to safely managed drinking water from South East Asian countries!

Use this tool to explore:
- **Predictions** based on population, year, water stress, and more
- **How the model works**
- **Data insights** from Southeast Asia based on UNICEF's data

Go to the **sidebar** and choose a page to get started!
""")
