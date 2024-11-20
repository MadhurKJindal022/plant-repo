import streamlit as st
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import plotly.graph_objects as go

# Load your trained leaf disease classification model (model3.h5)
model = load_model('model3.h5')

# Define the label dictionary for leaf classification
labels = {
    0: 'Apple', 1: 'Blueberry', 2: 'Cherry', 3: 'Corn', 4: 'Grape', 5: 'Orange',
    6: 'Peach', 7: 'Pepper', 8: 'Potato', 9: 'Raspberry', 10: 'Soybean', 11: 'Squash',
    12: 'Strawberry', 13: 'Tomato'
}

# Preprocess the image for model prediction
def preprocess_image(image, target_size=(225, 225)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Fetch plant details from a CSV file
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
        st.error("Error fetching data from CSV.")
    return None

# Fetch plant details from Google
def fetch_plant_details(plant_name):
    search_url = f"https://www.google.com/search?q={plant_name.replace(' ', '+')}+plant+details"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')
    plant_details = {"name": plant_name}

    try:
        plant_details.update({
            "scientific_name": soup.find('span', class_="LrzXr").text if soup.find('span', class_="LrzXr") else "N/A",
            "climate": "N/A", "life_span": "N/A", "soil": "N/A", "rainfall": "N/A", "regions": "N/A", "season": "N/A"
        })
    except Exception:
        st.error("Could not retrieve the plant details from Google.")
        return None

    return plant_details

# Streamlit layout for the app
st.title("Plant Health Assessment and Leaf Identification")

# Leaf classification section
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
    if st.button("Use Predicted Plant Name"):
        st.session_state['plant_name'] = predicted_label
        st.success(f"Predicted plant '{predicted_label}' copied to lookup section.")

# Plant information lookup section
st.subheader("Plant Information Lookup")
plant_name = st.text_input("Enter Plant Name", st.session_state.get('plant_name', "").strip())

if st.button("Get Plant Information"):
    if plant_name:
        plant_details = fetch_from_csv(plant_name) or fetch_plant_details(plant_name)
        if plant_details:
            st.success(f"Information for **{plant_details['name']}**:")
            st.table({
                "Attribute": ["Name", "Scientific Name", "Climate of Growth", "Life Span", "Soil", "Adequate Rainfall", "Top 3 Regions", "Season"],
                "Details": [plant_details['name'], plant_details['scientific_name'], plant_details['climate'], plant_details['life_span'], plant_details['soil'], plant_details['rainfall'], plant_details['regions'], plant_details['season']]
            })
        else:
            st.warning("No information found.")
    else:
        st.error("Please enter a plant name.")

# Section for plant health assessment
st.subheader("Plant Health Assessment")
col1, col2 = st.columns(2)

# Define plant health parameters and inputs
parameters = ["Sunlight (hours)", "Nutrients (NPK)", "Water (liters)", "Humidity (%)", "Soil pH", "CO2 (ppm)"]
max_values = [12, 100, 10, 100, 14, 1000]

# Collect input for each parameter
with col1:
    sunlight = st.slider("Sunlight (hours)", 0, 12, 6)
    nutrients_n = st.slider("Nitrogen (N)", 0, 100, 50)
    water = st.slider("Water (liters)", 0, 10, 5)
    co2 = st.slider("CO2 (ppm)", 300, 1000, 600)

with col2:
    nutrients_p = st.slider("Phosphorus (P)", 0, 100, 50)
    nutrients_k = st.slider("Potassium (K)", 0, 100, 50)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    soil_ph = st.slider("Soil pH", 0, 14, 7)

# Calculate normalized inputs and average health score
inputs = [sunlight, (nutrients_n + nutrients_p + nutrients_k) / 3, water, humidity, soil_ph, co2]
normalized_inputs = [x / max_val for x, max_val in zip(inputs, max_values)]
average_health_score = np.mean(normalized_inputs)

# Health assessment message
if average_health_score > 0.75:
    st.success("Plant is Healthy")
elif 0.5 < average_health_score <= 0.75:
    st.warning("Plant shows minor faults or chance of disease")
else:
    st.error("Plant is in danger")

# Options to select visualization type
st.subheader("Choose Visualization Type")
viz_option = st.radio("Select chart type", ("Radar Chart", "Bar Chart", "Table"))

# Number of days to observe recovery and calculate suggested values
days = st.slider("Number of days to observe recovery", 1, 30, 15)
ideal_values = [12, 100, 10, 100, 7, 400]
recovery_factor = np.linspace(0, 1, days)
suggested_values = [x * recovery_factor[-1] + (1 - recovery_factor[-1]) * y for x, y in zip(ideal_values, inputs)]
normalized_suggested = [x / max_val for x, max_val in zip(suggested_values, max_values)]

# Generate chart based on selection
if st.button("Generate Visualization"):
    if viz_option == "Radar Chart":
        # Radar chart (spider chart)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=normalized_inputs,
            theta=parameters,
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.3)' if average_health_score > 0.75 else 'rgba(255, 255, 0, 0.3)' if average_health_score > 0.5 else 'rgba(255, 0, 0, 0.3)',
            line_color='rgba(0, 255, 0, 0.3)' if average_health_score > 0.75 else 'rgba(255, 255, 0, 0.3)' if average_health_score > 0.5 else 'rgba(255, 0, 0, 0.3)',
            name='Plant Health'
        ))
        fig.add_trace(go.Scatterpolar(
            r=normalized_suggested,
            theta=parameters,
            fill=None,
            line=dict(color='rgba(0, 0, 255, 0.5)', dash='dash'),
            name=f'Suggested Recovery over {days} days'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Plant Health Radar Chart with Recovery Suggestion")
        st.plotly_chart(fig)

    elif viz_option == "Bar Chart":
        # Bar chart
        fig_bar = go.Figure()
        normalized_inputs_bar = [x / max_val for x, max_val in zip(inputs, max_values)]
        normalized_suggested_bar = [x / max_val for x, max_val in zip(suggested_values, max_values)]

        fig_bar.add_trace(go.Bar(
            x=parameters,
            y=normalized_inputs_bar,
            name='Current Values',
            marker_color='rgba(255, 99, 71, 0.6)',
        ))
        fig_bar.add_trace(go.Bar(
            x=parameters,
            y=normalized_suggested_bar,
            name=f'Suggested Values after {days} days',
            marker_color='rgba(0, 123, 255, 0.6)',
        ))
        fig_bar.update_layout(
            barmode='group',
            title='Current vs Suggested Recovery Values for Plant Parameters',
            xaxis_title='Parameters',
            yaxis_title='Normalized Value (0-1)',
            yaxis=dict(range=[0, 1]),
            showlegend=True
        )
        st.plotly_chart(fig_bar)

    elif viz_option == "Table":
        # Table with comparison data
        percentage_change = [((s - o) / o * 100) if o != 0 else 0 for s, o in zip(suggested_values, inputs)]
        data = {
            "Parameters": parameters,
            "Original Data": inputs,
            f"Suggested Data (after {days} days)": suggested_values,
            "Percentage Change (%)": percentage_change
        }
        df = pd.DataFrame(data)
        st.subheader("Comparison Table: Original vs Suggested Data")
        st.dataframe(df)
