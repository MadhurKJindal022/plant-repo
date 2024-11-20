import streamlit as st
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import time
import plotly.graph_objects as go

# Load the trained leaf disease classification model
model_plant = load_model('model3.h5')
model_disease = load_model('model.h5')
# Define label dictionary for leaf classification
plant_labels = {
    0: 'Apple', 1: 'Blueberry', 2: 'Cherry', 3: 'Corn', 4: 'Grape', 5: 'Orange', 
    6: 'Peach', 7: 'Pepper', 8: 'Potato', 9: 'Raspberry', 10: 'Soybean', 11: 'Squash', 
    12: 'Strawberry', 13: 'Tomato'
}

disease_labels = {
    0: 'Healthy', 1: 'Powdery', 2: 'Rust'
}
# CSS for animations and custom styling
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #FF6347;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); color: #FF4500; }
        100% { transform: scale(1); }
    }
    .stButton>button {
        color: white;
        font-size: 18px;
        background-color: #4CAF50;
        border-radius: 10px;
        padding: 8px 16px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .info {
        font-size: 20px;
        color: #4CAF50;
    }
    .highlighted {
        animation: colorChange 2s infinite alternate;
    }
    @keyframes colorChange {
        from { color: #4CAF50; }
        to { color: #FF6347; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

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


# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["🌿 Plant Detection", "🔍 Plant Information Lookup", "📊 Statistics", "🩺 Disease Detection"])

# Tab 1: Plant Detection
with tab1:
    st.markdown("<h1 class='title'>🌱 Plant Detection</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a leaf image...", type="jpg")
    

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Display a loading spinner during prediction
        with st.spinner("Classifying leaf..."):
            time.sleep(2)
            img_array = img_to_array(image.resize((225, 225)))
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Plant classification
            plant_predictions = model_plant.predict(img_array)
            predicted_plant_class = np.argmax(plant_predictions[0])
            predicted_plant_label = plant_labels[predicted_plant_class]

            # Disease classification
            disease_predictions = model_disease.predict(img_array)
            predicted_disease_class = np.argmax(disease_predictions[0])
            predicted_disease_label = disease_labels[predicted_disease_class]

        # Display the predictions
        st.markdown(f"<p class='info highlighted'>Prediction: <strong>{predicted_plant_label}</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p class='info highlighted'>Disease Status: <strong>{predicted_disease_label}</strong></p>", unsafe_allow_html=True)

        # Button to move predicted label to lookup section
        if st.button("Use Prediction for Lookup"):
            st.session_state.plant_name = predicted_plant_label
            st.success(f"Predicted plant name '{predicted_plant_label}' copied to lookup section.")
# Tab 2: Plant Information Lookup
with tab2:
    st.subheader("🔍 Plant Information Lookup")
    plant_name = st.text_input("Enter Plant Name", st.session_state.get('plant_name', "").strip())

    if st.button("Get Plant Information"):
        with st.spinner("Fetching plant details..."):
            time.sleep(1.5)
            if plant_name:
                plant_details = pd.read_csv('pla.csv')
                details = plant_details[plant_details['Name'].str.lower() == plant_name.lower()]

                if not details.empty:
                    st.success(f"Information for **{details['Name'].values[0]}** found in Database:")
                    st.table({
                        "Attribute": [
                            "Name", "Scientific Name", "Climate of Growth",
                            "Life Span", "Soil", "Adequate Rainfall", "Top 3 Regions", "Season"
                        ],
                        "Details": [
                            details['Name'].values[0], details['Scientific Name'].values[0], details['Climate of Growth'].values[0],
                            details['Life Span'].values[0], details['Soil'].values[0], details['Adequate Rainfall'].values[0],
                            f"{details['Regions1'].values[0]}, {details['Regions2'].values[0]}, {details['Regions3'].values[0]}", details['Season'].values[0]
                        ]
                    })
                else:
                    # Fallback to online search if plant not in CSV
                    online_info = fetch_plant_details(plant_name)
                    if online_info:
                        st.success(f"Information for **{online_info['name']}** found online:")
                        st.write(online_info)
                    else:
                        st.error("No information found for this plant.")
            else:
                st.error("Please enter a plant name to get information.")

# Tab 3: Statistics
with tab3:
    st.header("📊 Plant Health Assessment")
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

    # Calculate health score
    max_values = [12, 100, 10, 100, 14, 1000]
    inputs = [sunlight, (nutrients_n + nutrients_p + nutrients_k) / 3, water, humidity, soil_ph, co2]
    normalized_inputs = [x / max_val for x, max_val in zip(inputs, max_values)]
    average_health_score = np.mean(normalized_inputs)

    # Health assessment feedback with animated color
    if average_health_score > 0.75:
        chart_color = 'rgba(0, 255, 0, 0.3)'
        st.success("Plant is Healthy")
    elif 0.5 < average_health_score <= 0.75:
        chart_color = 'rgba(255, 255, 0, 0.3)'
        st.warning("Plant shows minor faults or chance of disease")
    else:
        chart_color = 'rgba(255, 0, 0, 0.3)'
        st.error("Plant is in danger")

    # Visualization (Radar and Bar Charts)
    ideal_values = [12, 100, 10, 100, 7, 400]
    normalized_suggested = [x / max_val for x, max_val in zip(ideal_values, max_values)]

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
        name='Suggested Ideal'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Dynamic Plant Health Radar Chart with Suggested Ideal")
    st.plotly_chart(fig)

    # Bar chart comparing current vs suggested values
    fig_bar = go.Figure()
    normalized_inputs_bar = [x / max_val for x, max_val in zip(inputs, max_values)]
    normalized_suggested_bar = [x / max_val for x, max_val in zip(ideal_values, max_values)]

    fig_bar.add_trace(go.Bar(
        x=parameters,
        y=normalized_inputs_bar,
        name='Current Values',
        marker_color='rgba(255, 99, 71, 0.6)',
    ))
    fig_bar.add_trace(go.Bar(
        x=parameters,
        y=normalized_suggested_bar,
        name='Suggested Ideal Values',
        marker_color='rgba(0, 123, 255, 0.6)',
    ))
    fig_bar.update_layout(barmode='group', title='Dynamic Comparison: Current vs Suggested Ideal Values for Plant Parameters', xaxis_title='Parameters', yaxis_title='Normalized Value (0-1)', yaxis=dict(range=[0, 1]), showlegend=True)
    st.plotly_chart(fig_bar)

    # Create a DataFrame to display the comparison table
    data = {
        "Parameters": parameters,
        "Original Data": inputs,
        "Normalized Data (0-1)": normalized_inputs_bar,
        "Suggested Ideal Data": ideal_values,
        "Suggested Normalized Data (0-1)": normalized_suggested_bar
    }
    df = pd.DataFrame(data)

    st.write("Comparison of parameters and suggested ideal values:")
    st.table(df)

with tab4:
    st.markdown("<h1 class='title'>🌿 Disease Detection</h1>", unsafe_allow_html=True)
    
    # Input for plant name
    plant_name_input = st.text_input("Plant Name for Disease Detection", st.session_state.get('plant_name', ""), key="disease_detection_input")

    if plant_name_input:
        # Load the plant diseases CSV
        plant_diseases = pd.read_csv('plant_diseases.csv')

        # Filter diseases for the entered plant
        plant_disease_details = plant_diseases[plant_diseases['Plant'].str.lower() == plant_name_input.lower()]

        if not plant_disease_details.empty:
            st.success(f"Disease Information for **{plant_name_input.capitalize()}**:")

            # Display plant diseases details in a table
            for _, disease_row in plant_disease_details.iterrows():
                # Animated disease details
                st.markdown(f"""
                    <div class="highlighted">
                        <h3>{disease_row['Possible Diseases']}</h3>
                        <p><strong>Reason for Disease:</strong> {disease_row['Reason for Disease']}</p>
                        <p><strong>Prevention:</strong> {disease_row['Prevention']}</p>
                        <p><strong>Suggestion:</strong> {disease_row['Suggestion']}</p>
                        <p><strong>Additional Information:</strong> {disease_row['Info Over the Disease']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
            # Optional: Add an animated alert to grab attention
            st.markdown(f"""
                <div style="background-color: #FF6347; color: white; padding: 10px; border-radius: 5px; animation: pulse 2s infinite;">
                    <strong>Warning:</strong> This disease could significantly impact plant health. Take necessary precautions!
                </div>
            """, unsafe_allow_html=True)

        else:
            st.error(f"No disease information found for {plant_name_input.capitalize()}. Please ensure the plant name is correct.")
