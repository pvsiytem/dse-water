import streamlit as st
import random

fun_facts = {
    "Water": [
        "💧 Water covers about 71% of Earth's surface.",
        "🚿 It takes about 2,641 gallons of water to produce a single pair of jeans.",
        "🧊 Ice is less dense than liquid water.",
        "🧪 Pure water has a neutral pH of 7.",
        "🌍 Nearly 97% of the world’s water is salty or otherwise undrinkable.",
        "🏞️ The human brain is about 75% water.",
        "🔁 Water is the only substance that naturally exists in all three states: solid, liquid, and gas.",
        "🚰 An average person in the U.S. uses 80–100 gallons of water per day.",
        "🧊 Antarctica holds about 90% of the world’s fresh water as ice.",
        "🌦️ Water can dissolve more substances than any other liquid."
    "Sea": [
        "🌊 The Mariana Trench is the deepest part of the world's oceans, about 11,000 meters!",
        "🐙 The giant Pacific octopus can have an arm span of up to 16 feet!",
        "🌐 94% of Earth’s living species exist within the oceans.",
        "🔱 The sea is home to the longest-living mammal, the bowhead whale which can live over 200 years.",
        "🦑 Some squids have eyes as big as basketballs.",
        "🌡️ The ocean absorbs about 90% of the heat from global warming.",
        "🧭 Earth’s oceans contain enough salt to cover all the land with a 500-foot-deep layer.",
        "💡 Some deep-sea creatures produce their own light, a phenomenon called bioluminescence.",
        "🐬 Dolphins call each other by unique names via whistle patterns.",
        "🧜‍♀️ The pressure at the deepest parts of the sea is over 1,000 times the air pressure at sea level."
    ],
    "South East Asia": [
        "Brunei Darussalam is also the only country with an absolute monarchy system left in South East Asia.",
        "Cambodia is home to the Asian giant softshell turtle and the critically endangered Irrawaddy dolphins.",
        "Indonesia is made up of 18,307 islands.",
        "Laos is known as the Land of a Million Elephants.",
        "Malaysia is home to Mount Kinabalu, the tallest peak in the country, standing at 13,455 feet (4,101 metres) above sea level.",
        "Myanmar is one of three countries that have not yet applied the International System of Units (the metric system).",
        "Philippines is the second largest producer of coconut products in the world.",
        "Singapore is one of the only three modern city-states exist in the world, other than Monaco and the Vatican.",
        "Thailand is home to the smallest mammal in the world as well as the largest fish in the world, which are the Kitti’s Hog-Nosed Bat and the Whale Shark respectively.",
        "East Timor (Timor Leste)  is a top destination for whale and dolphin watching as well as diving due to its location in the Coral Triangle.",
        "Vietnam is a nation of rivers with over 2,360 rivers flowing through its landscapes.",
    ],
}


st.markdown("### 🎉 Want a fun fact?")
fact_type = st.selectbox("Choose your category:", ["Water", "Sea", "South East Asia"])
if st.button("💡 Show me a fun fact!"):
    selected_fact = random.choice(fun_facts[fact_type])
    st.toast(selected_fact, icon="📣")
