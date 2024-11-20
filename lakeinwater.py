import streamlit as st
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import plotly.graph_objects as go
from io import BytesIO
from reportlab.pdfgen import canvas

# Check if the model loads correctly
try:
    model = load_model('model3.h5')
    st.write("Model loaded successfully.")
except Exception as e:
    st.error("Model loading failed.")
    st.write(e)

# Define label dictionary for leaf classification
labels = {
    0: 'Apple', 1: 'Blueberry', 2: 'Cherry', 3: 'Corn', 4: 'Grape', 5: 'Orange', 
    6: 'Peach', 7: 'Pepper', 8: 'Potato', 9: 'Raspberry', 10: 'Soybean', 11: 'Squash', 
    12: 'Strawberry', 13: 'Tomato'
}

# Initialize session state for plant_name if it doesn’t exist
if 'plant_name' not in st.session_state:
    st.session_state.plant_name = ""

def preprocess_image(image, target_size=(225, 225)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def fetch_plant_details(plant_name):
    search_url = f"https://www.google.com/search?q={plant_name.replace(' ', '+')}+plant+details"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')
    plant_details = {
        'name': plant_name,
        'scientific_name': soup.find('span', class_="LrzXr").text if soup.find('span', class_="LrzXr") else "N/A",
        'climate': "N/A", 'life_span': "N/A", 'soil': "N/A",
        'rainfall': "N/A", 'regions': "N/A", 'season': "N/A"
    }
    return plant_details

def fetch_from_csv(plant_name):
    try:
        df = pd.read_csv('pla.csv')
        plant_details = df[df['Name'].str.lower() == plant_name.lower()]
        if not plant_details.empty:
            return {
                "name": plant_details['Name'].values[0],
                "scientific_name": plant_details['Scientific Name'].values[0],
                "climate": plant_details['Climate of Growth'].values[0],
                "life_span": plant_details['Life Span'].values[0],
                "soil": plant_details['Soil'].values[0],
                "rainfall": plant_details['Adequate Rainfall'].values[0],
                "regions": f"{plant_details['Regions1'].values[0]}, {plant_details['Regions2'].values[0]}, {plant_details['Regions3'].values[0]}",
                "season": plant_details['Season'].values[0]
            }
    except Exception as e:
        st.write("Error reading CSV:", e)
    return None

st.title("Plant Health Assessment and Leaf Identification")

# Leaf Disease Classification
st.subheader("Plant Leaf Classification")
uploaded_file = st.file_uploader("Upload a leaf image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = labels[predicted_class]
    st.write(f"Prediction: {predicted_label}")

    if st.button("Image Info"):
        st.session_state.plant_name = predicted_label
        st.success(f"Predicted plant name '{predicted_label}' copied to lookup section.")

# Plant Information Lookup
st.subheader("Plant Information Lookup")
plant_name = st.text_input("Enter Plant Name", st.session_state.get('plant_name', "").strip())

if st.button("Get Plant Information"):
    if plant_name:
        plant_details = fetch_from_csv(plant_name)
        if plant_details:
            st.success(f"Information for **{plant_details['name']}** found in Database:")
            st.write(plant_details)
        else:
            st.warning("No information found in Database. Searching online...")
            online_info = fetch_plant_details(plant_name)
            if online_info:
                st.success(f"Information for **{online_info['name']}** found online:")
                st.write(online_info)
            else:
                st.error("No information found.")
    else:
        st.error("Please enter a plant name.")

# Plant Health Assessment
st.subheader("Plant Health Assessment")
parameters = ["Sunlight (hours)", "Nutrients (NPK)", "Water (liters)", "Humidity (%)", "Soil pH", "CO2 (ppm)"]
sunlight = st.slider("Sunlight (hours)", 0, 12, 6)
nutrients_n = st.slider("Nitrogen (N)", 0, 100, 50)
water = st.slider("Water (liters)", 0, 10, 5)
co2 = st.slider("CO2 (ppm)", 300, 1000, 600)
nutrients_p = st.slider("Phosphorus (P)", 0, 100, 50)
nutrients_k = st.slider("Potassium (K)", 0, 100, 50)
humidity = st.slider("Humidity (%)", 0, 100, 50)
soil_ph = st.slider("Soil pH", 0, 14, 7)

max_values = [12, 100, 10, 100, 14, 1000]
inputs = [sunlight, (nutrients_n + nutrients_p + nutrients_k) / 3, water, humidity, soil_ph, co2]
normalized_inputs = [x / max_val for x, max_val in zip(inputs, max_values)]
average_health_score = np.mean(normalized_inputs)

if average_health_score > 0.75:
    st.success("Plant is Healthy")
elif 0.5 < average_health_score <= 0.75:
    st.warning("Plant shows minor faults or chance of disease")
else:
    st.error("Plant is in danger")

# Download Report as PDF
if st.button("Download Report as PDF"):
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer)
    c.drawString(100, 800, "Plant Health Assessment and Leaf Identification Report")
    c.drawString(100, 780, f"Predicted Plant Label: {predicted_label}")
    c.drawString(100, 760, f"Health Score: {average_health_score:.2f}")
    c.showPage()
    c.save()
    
    st.download_button(
        label="Download PDF",
        data=pdf_buffer.getvalue(),
        file_name="plant_health_report.pdf",
        mime="application/pdf"
    )
