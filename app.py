import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and columns
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

# --- App Header ---
st.set_page_config(page_title="Housing Price Prediction", page_icon="🏠", layout="centered")
st.title("🏠 **Housing Price Prediction App**")
st.markdown("""
Welcome to the **Housing Price Prediction App**!  
This interactive tool allows you to estimate housing prices based on various features.  
Use the sidebar to enter details about the house, and view your predicted price instantly! ✨
""")

# --- Sidebar for User Input ---
st.sidebar.header("📝 Enter House Details")

def user_input_features():
    area = st.sidebar.slider('🏠 Area (sq ft)', 1000, 10000, 5000)
    bedrooms = st.sidebar.slider('🛏️ Bedrooms', 1, 5, 3)
    bathrooms = st.sidebar.slider('🛁 Bathrooms', 1, 4, 2)
    stories = st.sidebar.slider('🏢 Stories', 1, 4, 2)
    parking = st.sidebar.slider('🚗 Parking spaces', 0, 3, 1)
    mainroad = st.sidebar.selectbox('🚦 Main Road?', ('yes', 'no'))
    guestroom = st.sidebar.selectbox('🛋️ Guest Room?', ('yes', 'no'))
    basement = st.sidebar.selectbox('🏚️ Basement?', ('yes', 'no'))
    hotwaterheating = st.sidebar.selectbox('🔥 Hot Water Heating?', ('yes', 'no'))
    airconditioning = st.sidebar.selectbox('❄️ Air Conditioning?', ('yes', 'no'))
    prefarea = st.sidebar.selectbox('🌳 Preferred Area?', ('yes', 'no'))
    furnishingstatus = st.sidebar.selectbox('🪑 Furnishing Status', 
                                            ('unfurnished', 'semi-furnished', 'furnished'))
    
    data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus
    }
    
    return pd.DataFrame([data])

input_df = user_input_features()

# --- Display User Input ---
with st.expander("🔎 **View Your Input Data**"):
    st.write(input_df)

# --- Data Preprocessing ---
input_df = pd.get_dummies(input_df, drop_first=True)

# Ensure all columns present
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Align with model columns
input_df = input_df[model_columns]

# Scale numeric features
numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# --- Make Prediction ---
prediction = model.predict(input_df)[0]

# --- Result Display ---
st.markdown("""
<hr style="border: 1px solid #f0f0f0;">
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h2 style='text-align: center; color: #2E8B57;'>💰 Predicted Price</h2>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #FF5733;'>₹ {prediction:,.2f}</h1>", unsafe_allow_html=True)
    st.success("Prediction complete! 🎉")

st.markdown("""
<hr style="border: 1px solid #f0f0f0;">
""", unsafe_allow_html=True)

st.caption("🔗 Created as part of the Introduction to Data Science course project.")
