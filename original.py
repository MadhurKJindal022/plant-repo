import streamlit as st
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import plotly.graph_objects as go

# Load your trained leaf disease classification model (model2.h5)
model = load_model('model3.h5')

# Define the label dictionary for leaf classification
labels = {
    0: 'Apple', 1: 'Blueberry', 2: 'Cherry', 3: 'Corn', 4: 'Grape', 5: 'Orange', 
    6: 'Peach', 7: 'Pepper', 8: 'Potato', 9: 'Raspberry', 10: 'Soybean', 11: 'Squash', 
    12: 'Strawberry', 13: 'Tomato'
}

# Function to preprocess the image
def preprocess_image(image, target_size=(225, 225)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to fetch plant details from Google search
def fetch_plant_details(plant_name):
    search_url = f"https://www.google.com/search?q={plant_name.replace(' ', '+')}+plant+details"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')

    plant_details = {}

    try:
        plant_details['name'] = plant_name
        plant_details['scientific_name'] = soup.find('span', class_="LrzXr").text if soup.find('span', class_="LrzXr") else "N/A"
        plant_details['climate'] = "N/A"
        plant_details['life_span'] = "N/A"
        plant_details['soil'] = "N/A"
        plant_details['rainfall'] = "N/A"
        plant_details['regions'] = "N/A"
        plant_details['season'] = "N/A"
    except Exception as e:
        st.error("Could not retrieve the plant details from Google. Please try a different plant.")
        return None

    return plant_details

# Function to fetch plant details from CSV file
def fetch_from_csv(plant_name):
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
    else:
        return None

# Streamlit app layout
st.title("Plant Health Assessment and Leaf Identification")

# Section for leaf disease classification
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
        st.session_state.plant_name = predicted_label  # Store predicted label in session state
        st.success(f"Predicted plant name '{predicted_label}' copied to lookup section.")

# Section for plant information lookup
st.subheader("Plant Information Lookup")
plant_name = st.text_input("Enter Plant Name", st.session_state.get('plant_name', "").strip())  # Pre-fill with predicted label

if st.button("Get Plant Information"):
    if plant_name:
        plant_details = fetch_from_csv(plant_name)

        if plant_details:
            st.success(f"Information for **{plant_details['name']}** found in Database:")
            st.table({
                "Attribute": [
                    "Name", "Scientific Name", "Climate of Growth",
                    "Life Span", "Soil", "Adequate Rainfall", "Top 3 Regions", "Season"
                ],
                "Details": [
                    plant_details['name'], plant_details['scientific_name'], plant_details['climate'],
                    plant_details['life_span'], plant_details['soil'], plant_details['rainfall'], plant_details['regions'], plant_details['season']
                ]
            })

        else:
            st.warning("No information found for the specified plant in the Database. Searching online...")
            # Call the online fetch function if not found in CSV
            online_info = fetch_plant_details(plant_name)
            if online_info:
                st.success(f"Information for **{online_info['name']}** found online:")
                st.write(online_info)
            else:
                st.error("No information found for this plant.")

    else:
        st.error("Please enter a plant name to get information.")
# Section for plant health assessment (this remains unchanged)
st.subheader("Plant Health Assessment")
col1, col2 = st.columns(2)

parameters = ["Sunlight (hours)", "Nutrients (NPK)", "Water (liters)", "Humidity (%)", "Soil pH", "CO2 (ppm)"]

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

max_values = [12, 100, 10, 100, 14, 1000]
inputs = [sunlight, (nutrients_n + nutrients_p + nutrients_k) / 3, water, humidity, soil_ph, co2]
normalized_inputs = [x / max_val for x, max_val in zip(inputs, max_values)]

average_health_score = np.mean(normalized_inputs)

if average_health_score > 0.75:
    chart_color = 'rgba(0, 255, 0, 0.3)'
    st.success("Plant is Healthy")
elif 0.5 < average_health_score <= 0.75:
    chart_color = 'rgba(255, 255, 0, 0.3)'
    st.warning("Plant shows minor faults or chance of disease")
else:
    chart_color = 'rgba(255, 0, 0, 0.3)'
    st.error("Plant is in danger")

days = st.slider("Number of days to observe recovery", 1, 30, 15)
ideal_values = [12, 100, 10, 100, 7, 400]
recovery_factor = np.linspace(0, 1, days)
suggested_values = [x * recovery_factor[-1] + (1 - recovery_factor[-1]) * y for x, y in zip(ideal_values, inputs)]
normalized_suggested = [x / max_val for x, max_val in zip(suggested_values, max_values)]

# Radar chart
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=normalized_inputs,
    theta=parameters,
    fill='toself',
    fillcolor=chart_color,
    line_color=chart_color,
    name='Plant Health'
))
fig.add_trace(go.Scatterpolar(
    r=normalized_suggested,
    theta=parameters,
    fill=None,
    line=dict(color='rgba(0, 0, 255, 0.5)', dash='dash'),
    name=f'Suggested Recovery over {days} days'
))
fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Dynamic Plant Health Radar Chart with Recovery Suggestion")
st.plotly_chart(fig)

# Bar chart comparing current vs suggested values
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
fig_bar.update_layout(barmode='group', title='Dynamic Comparison: Current vs Suggested Recovery Values for Plant Parameters', xaxis_title='Parameters', yaxis_title='Normalized Value (0-1)', yaxis=dict(range=[0, 1]), showlegend=True)
st.plotly_chart(fig_bar)

# Calculate percentage change for each parameter
percentage_change = [((s - o) / o * 100) if o != 0 else 0 for s, o in zip(suggested_values, inputs)]

# Create a DataFrame to display the comparison table
data = {
    "Parameters": parameters,
    "Original Data": inputs,
    f"Suggested Data (after {days} days)": suggested_values,
    "Percentage Change (%)": percentage_change
}
df = pd.DataFrame(data)
st.subheader("Comparison Table: Original vs Suggested Data")
st.dataframe(df)



