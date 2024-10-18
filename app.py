import streamlit as st
import pandas as pd
import joblib
import time

# Set layout
st.set_page_config(layout="wide")

# Adding JavaScript to dynamically change button color
st.markdown("""
    <style>
    .stApp {
        background-color: #1e1e1e;
    }
    h1, h2 {
        color: white;
    }
    .stButton > button {
        background-color: #4CAF50 !important;  /* Default green background */
        color: white !important;  /* White text by default */
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease, color 0.3s ease;  /* Smooth transitions */
    }
    .stButton > button:hover {
        background-color: #45a049 !important;  /* Darker green on hover */
    }
    .stButton:active > button {
        background-color: #007BFF !important;  /* Blue background on click */
    }
    .stButton:focus > button {
        background-color: #4CAF50 !important;  /* Green background after clicking */
    }
    .prediction-result {
        font-size: 36px;
        font-weight: bold;
        color: #f5a623;
        text-align: center;
        margin-top: 20px;
    }
    .probability-result {
        font-size: 28px;
        color: #1f77b4;
        text-align: center;
        margin-top: 10px;
    }
    </style>

    <script>
    // Function to change button color to blue on click
    document.querySelectorAll('.stButton button').forEach((el) => {
        el.addEventListener('mousedown', function() {
            this.style.backgroundColor = '#007BFF';  // Change to blue on click
        });
        el.addEventListener('mouseup', function() {
            setTimeout(() => {
                this.style.backgroundColor = '#4CAF50';  // Return to green after click
            }, 200);  // Delay for a smooth transition
        });
    });
    </script>
    """, unsafe_allow_html=True)

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('models/aussie_rain_pipeline.joblib')

model = load_model()

# Load the dataset
@st.cache_resource
def load_data():
    return pd.read_csv('data/weatherAUS.csv')

data = load_data()

# Calculate min and max values from your dataset
min_temp_min = data['MinTemp'].min()
min_temp_max = data['MinTemp'].max()

max_temp_min = data['MaxTemp'].min()
max_temp_max = data['MaxTemp'].max()

rainfall_min = data['Rainfall'].min()
rainfall_max = data['Rainfall'].max()

evaporation_min = data['Evaporation'].min()
evaporation_max = data['Evaporation'].max()

sunshine_min = data['Sunshine'].min()
sunshine_max = data['Sunshine'].max()

cloud_3pm_min = data['Cloud3pm'].min()
cloud_3pm_max = data['Cloud3pm'].max()

temp_9am_min = data['Temp9am'].min()
temp_9am_max = data['Temp9am'].max()

temp_3pm_min = data['Temp3pm'].min()
temp_3pm_max = data['Temp3pm'].max()

def predict_weather(input_data):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1][0]
    return prediction[0], probability

# Header and intro
#st.image('image/logo.png', width=100)
st.title('🌦️ Weather Prediction: Will it Rain Tomorrow?')
st.markdown("This app predicts whether it will rain tomorrow based on today's weather conditions.")
st.markdown("### Enter today's weather conditions")
st.divider()

# Input fields for weather data using min-max from dataset
st.header("Enter today's weather conditions:")

# Create columns to organize input fields
col1, col2, col3 = st.columns(3)

with col1:
    min_temp = st.number_input('Min Temperature (°C)', value=min_temp_min, min_value=min_temp_min, max_value=min_temp_max, step=0.1)
    max_temp = st.number_input('Max Temperature (°C)', value=max_temp_min, min_value=max_temp_min, max_value=max_temp_max, step=0.1)
    rainfall = st.number_input('Rainfall (mm)', value=rainfall_min, min_value=rainfall_min, max_value=rainfall_max, step=0.1)

with col2:
    evaporation = st.number_input('Evaporation (mm)', value=evaporation_min, min_value=evaporation_min, max_value=evaporation_max, step=0.1)
    sunshine = st.number_input('Sunshine (hrs)', value=sunshine_min, min_value=sunshine_min, max_value=sunshine_max, step=0.1)
    cloud_3pm = st.slider('Cloudiness at 3pm (oktas)', int(cloud_3pm_min), int(cloud_3pm_max), 4)

with col3:
    temp_9am = st.number_input('Temperature at 9am (°C)', value=temp_9am_min, min_value=temp_9am_min, max_value=temp_9am_max, step=0.1)
    temp_3pm = st.number_input('Temperature at 3pm (°C)', value=temp_3pm_min, min_value=temp_3pm_min, max_value=temp_3pm_max, step=0.1)
    rain_today = st.selectbox('Was it raining today?', ['Yes', 'No'])

# Add 'RainToday' to the input data
data = pd.DataFrame({
    'MinTemp': [min_temp],
    'MaxTemp': [max_temp],
    'Rainfall': [rainfall],
    'Evaporation': [evaporation],
    'Sunshine': [sunshine],
    'Cloud3pm': [cloud_3pm],
    'Temp9am': [temp_9am],
    'Temp3pm': [temp_3pm],
    'RainToday': [rain_today]  # Pass 'Yes' or 'No' directly
})

# Button to predict with progress bar
if st.button("Predict Rain"):
    with st.spinner('Predicting...'):
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        prediction, probability = model.predict(data), model.predict_proba(data)[:, 1][0]

    # Displaying the result with improved formatting and centered text
    st.markdown(f"<div class='prediction-result'>Will it rain tomorrow? {'🌧️ Yes' if prediction == 'Yes' else '☀️ No'}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='probability-result'>Prediction Probability: {probability:.2f}</div>", unsafe_allow_html=True)
