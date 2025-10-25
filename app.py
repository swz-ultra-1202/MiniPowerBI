
import streamlit as st
import pandas as pd
import numpy as np

# =============================
# 🌟 PAGE CONFIGURATION
# =============================
st.set_page_config(
    page_title="🦾 Mini Power BI ⚙️",
    page_icon="🦾 ",
    layout="wide"
)

# =============================
# 🧠 SIDEBAR NAVIGATION
# =============================
st.sidebar.image("logo.png", width=100)
st.sidebar.title("🤖 AI/ML Data Intelligence Dashboard")

#  sidebar buttons
home_btn = st.sidebar.button("🏠 Home")
upload_btn = st.sidebar.button("📂 Upload Data")
clean_btn = st.sidebar.button("🧹 Clean Data")
visual_btn = st.sidebar.button("📊 Visualize Data")
predict_btn = st.sidebar.button("🤖 Predict with ML")
forecast_btn = st.sidebar.button("📈 Forecast Trends")
chat_btn = st.sidebar.button("💬 Chat with Data")
about_btn = st.sidebar.button("📄 About")

# =============================
# 🔄 PAGE NAVIGATION CONTROL
# =============================
# Store current page in session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

if home_btn:
    st.session_state.page = "Home"
elif upload_btn:
    st.session_state.page = "Upload"
elif clean_btn:
    st.session_state.page = "Clean"
elif visual_btn:
    st.session_state.page = "Visualize"
elif predict_btn:
    st.session_state.page = "Predict"
elif forecast_btn:
    st.session_state.page = "Forecast"
elif chat_btn:
    st.session_state.page = "Chat"
elif about_btn:
    st.session_state.page = "About"

# =============================
# 🧩 PAGE CONTENT SECTIONS
# =============================

if st.session_state.page == "Home":
    st.title("🧩 Mini Power BI 🧠")
    st.write("Welcome to your AI-powered analytics dashboard.")
    st.image("logo.png", width=250)
    st.markdown("""
    ### 🚀 Features:
    - Upload and clean your data  
    - Visualize insights interactively  
    - Train AI/ML models automatically  
    - Forecast trends  
    - Chat with your data using AI  
    """)
    st.success("Start by uploading your dataset → 📂 Upload Data")