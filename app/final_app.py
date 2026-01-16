import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"   # FIXED: wide layout for full table
)

# Load Model & Scaler
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("model/rf_model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_artifacts()

# App Title
st.title("🍷 Wine Quality Prediction App")
st.write(
    """
    Predict wine quality into **Average**, **Good**, or **Excellent**
    using physicochemical properties.
    """
)

st.markdown("---")

# Sidebar Inputs
st.sidebar.header("🧪 Physicochemical Properties")

fixed_acidity = st.sidebar.slider("Fixed Acidity", 3.0, 16.0, 7.0, 0.1)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.6, 0.4, 0.01)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.5, 0.3, 0.01)
residual_sugar = st.sidebar.slider("Residual Sugar", 0.5, 50.0, 6.0, 0.1)
chlorides = st.sidebar.slider("Chlorides", 0.01, 0.6, 0.05, 0.001)
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 1, 300, 30)
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 6, 500, 120)
density = st.sidebar.slider("Density", 0.985, 1.05, 0.995, 0.0001)
pH = st.sidebar.slider("pH", 2.7, 4.5, 3.2, 0.01)
sulphates = st.sidebar.slider("Sulphates", 0.2, 2.0, 0.6, 0.01)
alcohol = st.sidebar.slider("Alcohol (%)", 8.0, 15.0, 10.5, 0.1)

color = st.sidebar.radio("Wine Color", ["Red", "White"])
color_encoded = 0 if color == "Red" else 1

# Input DataFrame
input_data = pd.DataFrame({
    "fixed acidity": [fixed_acidity],
    "volatile acidity": [volatile_acidity],
    "citric acid": [citric_acid],
    "residual sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free sulfur dioxide": [free_sulfur_dioxide],
    "total sulfur dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol],
    "color": [color_encoded]
})

# Prediction
st.markdown("## 🔮 Prediction")

if st.button("Predict Wine Quality"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == "Excellent":
        st.success("🍾 **Excellent Quality Wine**")
    elif prediction == "Good":
        st.info("🍷 **Good Quality Wine**")
    else:
        st.warning("⚠️ **Average Quality Wine**")

    st.markdown("---")

    # Input Summary 
    st.subheader("📊 Input Summary")
    st.dataframe(input_data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Machine Learning Wine Quality Classifier | Random Forest</p>",
    unsafe_allow_html=True
)
