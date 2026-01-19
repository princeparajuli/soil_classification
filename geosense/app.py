"""
app.py - Integrated Field Inspection System
Combines Image Classification + Tabular Prediction
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['clay', 'gravel', 'sand', 'silt']
IMG_SIZE = 224

# Soil Engineering Properties Database
SOIL_PROPERTIES = {
    'clay': {
        'emoji': '🟫',
        'color': '#8B4513',
        'moisture_range': (20.0, 45.0),
        'bulk_density_range': (12.0, 18.0),
        'bearing_capacity': '75-200 kN/m²',
        'angle_of_friction': '0-20°',
        'cohesion': '25-100 kPa',
        'permeability': '10⁻⁸ to 10⁻⁹ cm/s',
        'foundation_type': 'Deep foundation (Piles/Piers)',
        'foundation_depth': '2.5-4.0 m',
        'uscs': 'CL/CH',
        'recommendations': [
            'Use pile foundation for heavy structures',
            'Provide adequate drainage to prevent swelling',
            'Consider soil stabilization with lime/cement',
            'Monitor moisture content during construction'
        ],
        'warnings': [
            '⚠️ High shrink-swell potential',
            '⚠️ Low bearing capacity when saturated',
            '⚠️ Prone to frost heave in cold climates'
        ]
    },
    'gravel': {
        'emoji': '⚪',
        'color': '#5C5C5C',
        'moisture_range': (2.0, 12.0),
        'bulk_density_range': (18.0, 22.0),
        'bearing_capacity': '300-600 kN/m²',
        'angle_of_friction': '35-45°',
        'cohesion': '0 kPa',
        'permeability': '10⁻¹ to 10⁻³ cm/s',
        'foundation_type': 'Shallow foundation (Spread footing)',
        'foundation_depth': '0.6-1.2 m',
        'uscs': 'GW/GP',
        'recommendations': [
            'Ideal for shallow foundations',
            'Excellent for road sub-base material',
            'Use for drainage layers and filters',
            'Minimal settlement expected'
        ],
        'warnings': [
            '✓ Excellent foundation material',
            '✓ No special precautions needed',
            '✓ High load-bearing capacity'
        ]
    },
    'sand': {
        'emoji': '🟡',
        'color': '#C9A961',
        'moisture_range': (8.0, 35.0),
        'bulk_density_range': (14.0, 20.0),
        'bearing_capacity': '150-300 kN/m²',
        'angle_of_friction': '28-35°',
        'cohesion': '0 kPa',
        'permeability': '10⁻³ to 10⁻⁵ cm/s',
        'foundation_type': 'Shallow foundation (Mat/Raft)',
        'foundation_depth': '1.0-2.0 m',
        'uscs': 'SW/SP',
        'recommendations': [
            'Use mat foundation for uniform load distribution',
            'Compact thoroughly with vibration',
            'Maintain water table below foundation',
            'Consider dynamic compaction for improvement'
        ],
        'warnings': [
            '⚠️ May liquefy during earthquakes if saturated',
            '⚠️ Wind erosion if not protected',
            '⚠️ Requires proper compaction'
        ]
    },
    'silt': {
        'emoji': '🟤',
        'color': '#8B7355',
        'moisture_range': (15.0, 40.0),
        'bulk_density_range': (13.0, 17.0),
        'bearing_capacity': '50-150 kN/m²',
        'angle_of_friction': '20-28°',
        'cohesion': '5-25 kPa',
        'permeability': '10⁻⁵ to 10⁻⁷ cm/s',
        'foundation_type': 'Deep foundation or ground improvement',
        'foundation_depth': '2.0-3.5 m',
        'uscs': 'ML/MH',
        'recommendations': [
            'Avoid shallow foundations if possible',
            'Consider ground improvement techniques',
            'Provide adequate drainage system',
            'Use geotextiles for reinforcement'
        ],
        'warnings': [
            '⚠️ Highly frost-susceptible',
            '⚠️ Low strength when saturated',
            '⚠️ Significant consolidation settlement'
        ]
    }
}

@st.cache_resource
def load_image_model():
    """Load the image classification model"""
    try:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, len(CLASS_NAMES))
        )
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"⚠️ Image model not loaded: {e}")
        return None

@st.cache_resource
def load_tabular_model():
    """Load the tabular prediction model"""
    try:
        with open('soil_bearing_model.pkl', 'rb') as f:
            models = pickle.load(f)
        return models
    except Exception as e:
        st.warning(f"⚠️ Tabular model not loaded: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def predict_image(model, image):
    """Predict soil type from image"""
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)
        return predicted.item(), confidence.item(), probs[0].cpu().numpy()

def predict_tabular(models, soil_type, moisture_content, bulk_density, 
                   grain_size, foundation_depth, field_observation_score):
    """Predict bearing capacity and safety class from tabular data"""
    bearing_model = models['bearing_capacity_model']
    safety_model = models['safety_class_model']
    scaler = models['scaler']
    le_soil = models['soil_type_encoder']
    le_grain = models['grain_size_encoder']
    le_safety = models['safety_class_encoder']
    
    # Encode categorical features
    soil_encoded = le_soil.transform([soil_type.capitalize()])[0]
    grain_encoded = le_grain.transform([grain_size])[0]
    
    # Create feature array
    features = np.array([[soil_encoded, moisture_content, bulk_density,
                         grain_encoded, foundation_depth, field_observation_score]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    bearing_pred = bearing_model.predict(features_scaled)[0]
    safety_pred_encoded = safety_model.predict(features_scaled)[0]
    safety_pred = le_safety.inverse_transform([safety_pred_encoded])[0]
    safety_proba = safety_model.predict_proba(features_scaled)[0]
    
    return {
        'bearing_capacity': round(bearing_pred, 2),
        'safety_class': safety_pred,
        'safety_probabilities': {
            class_name: round(prob * 100, 2)
            for class_name, prob in zip(le_safety.classes_, safety_proba)
        }
    }

def get_location_map(lat, lon, location_name):
    """Create folium map with location marker"""
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker(
        [lat, lon],
        popup=f"<b>Site Location</b><br>{location_name}",
        tooltip="Click for details",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    return m

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: #000000; padding: 1rem; }
    .stApp { background: #000000; }
    h1 {
        color: #FFFFFF; text-align: center; font-size: 3.5rem !important;
        font-weight: 900 !important; text-transform: uppercase;
        letter-spacing: 3px; margin-bottom: 0.5rem !important;
        text-shadow: 0 0 20px rgba(255,255,255,0.3);
    }
    .subtitle {
        text-align: center; color: #888888; font-size: 1.1rem;
        margin-bottom: 2rem; text-transform: uppercase;
        letter-spacing: 2px; font-weight: 400;
    }
    .inspection-card {
        background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
        border: 1px solid #2a2a2a; border-radius: 15px;
        padding: 2rem; margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.6);
    }
    .property-item {
        background: linear-gradient(145deg, #0a0a0a, #1a1a1a);
        border: 1px solid #2a2a2a; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .property-label {
        color: #888888; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .property-value {
        color: #FFFFFF; font-size: 1.3rem; font-weight: 700;
    }
    .warning-box {
        background: rgba(255, 69, 0, 0.1);
        border: 1px solid rgba(255, 69, 0, 0.3);
        border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
        color: #FF4500;
    }
    .success-box {
        background: rgba(0, 255, 127, 0.1);
        border: 1px solid rgba(0, 255, 127, 0.3);
        border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
        color: #00FF7F;
    }
    .recommendation-item {
        background: linear-gradient(145deg, #0a0a0a, #1a1a1a);
        border-left: 3px solid #FFFFFF;
        padding: 0.8rem 1rem; margin: 0.5rem 0;
        border-radius: 5px; color: #CCCCCC;
    }
    h3 {
        color: #FFFFFF !important; font-weight: 700 !important;
        text-transform: uppercase; letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Field Inspection System", page_icon="🔬", layout="wide")

# Header
st.markdown("<h1>🔬 FIELD INSPECTION SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Soil Analysis • Foundation Engineering • Site Investigation</p>", unsafe_allow_html=True)

# Load models
image_model = load_image_model()
tabular_model = load_tabular_model()

# Sidebar
with st.sidebar:
    st.markdown("### 📋 SITE INFORMATION")
    project_name = st.text_input("Project Name", "Project Site 2026")
    engineer_name = st.text_input("Engineer Name", "")
    test_date = st.date_input("Test Date", datetime.now())
    
    st.markdown("### 📍 LOCATION DATA")
    use_gps = st.checkbox("Use GPS Coordinates", value=False)
    
    if use_gps:
        latitude = st.number_input("Latitude", value=27.7172, format="%.6f")
        longitude = st.number_input("Longitude", value=85.3240, format="%.6f")
        location_input = f"{latitude}, {longitude}"
    else:
        location_input = st.text_input("Location/Address", "Kathmandu, Nepal")
        try:
            geolocator = Nominatim(user_agent="soil_inspector")
            location = geolocator.geocode(location_input)
            if location:
                latitude, longitude = location.latitude, location.longitude
            else:
                latitude, longitude = 27.7172, 85.3240
        except:
            latitude, longitude = 27.7172, 85.3240
    
    st.markdown("### 🏗️ SITE CONDITIONS")
    foundation_depth = st.slider("Foundation Depth (m)", 0.5, 5.0, 1.5, 0.1)
    water_table_depth = st.slider("Water Table Depth (m)", 0.5, 10.0, 3.0, 0.5)
    structure_type = st.selectbox("Structure Type", 
        ["Residential", "Commercial", "Industrial", "Bridge"])

# Main tabs
tab1, tab2, tab3 = st.tabs(["📸 IMAGE ANALYSIS", "📊 MANUAL ENTRY", "📄 REPORT"])

with tab1:
    st.markdown("### 📸 Soil Image Classification + Property Prediction")
    
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        camera_image = st.camera_input("📷 Capture soil sample")
    
    with col_upload2:
        uploaded_file = st.file_uploader("📁 Upload image", type=['jpg', 'jpeg', 'png','jfif'])
    
    # Process either camera or uploaded image
    input_image = camera_image if camera_image else uploaded_file
    
    if input_image and image_model:
        image = Image.open(input_image).convert('RGB')
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("<div class='inspection-card'>", unsafe_allow_html=True)
            st.markdown("### 🔬 CAPTURED SPECIMEN")
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Image classification
        processed_image = preprocess_image(image)
        predicted_class, confidence, all_probs = predict_image(image_model, processed_image)
        predicted_soil = CLASS_NAMES[predicted_class]
        soil_data = SOIL_PROPERTIES[predicted_soil]
        
        with col2:
            st.markdown(f"""
            <div class='inspection-card'>
                <div style='text-align: center;'>
                    <div style='font-size: 5rem;'>{soil_data['emoji']}</div>
                    <div style='font-size: 3rem; font-weight: 900; color: {soil_data["color"]}; text-transform: uppercase; margin: 1rem 0;'>
                        {predicted_soil.upper()}
                    </div>
                    <div style='background: #FFFFFF; color: #000000; padding: 0.8rem 2rem; border-radius: 50px; display: inline-block; font-weight: 700; font-size: 1.3rem;'>
                        {confidence*100:.1f}% CONFIDENCE
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Now get detailed predictions from tabular model
        if tabular_model:
            st.markdown("### ⚙️ DETAILED SOIL PROPERTY PREDICTION")
            
            # Use average values from soil type
            moisture_avg = np.mean(soil_data['moisture_range'])
            density_avg = np.mean(soil_data['bulk_density_range'])
            
            col_input1, col_input2, col_input3 = st.columns(3)
            
            with col_input1:
                moisture_content = st.number_input(
                    "Moisture Content (%)", 
                    min_value=0.0, max_value=100.0, 
                    value=float(moisture_avg),
                    step=0.1
                )
            
            with col_input2:
                bulk_density = st.number_input(
                    "Bulk Density (kN/m³)", 
                    min_value=10.0, max_value=25.0, 
                    value=float(density_avg),
                    step=0.1
                )
            
            with col_input3:
                grain_size = st.selectbox("Grain Size", ["Fine", "Medium", "Coarse"], key="grain_auto")
            
            field_score = st.slider("Field Observation Score (1-5)", 1, 5, 3)
            
            if st.button("🔍 Analyze Soil Properties", type="primary"):
                # Get tabular predictions
                prediction = predict_tabular(
                    tabular_model, predicted_soil, moisture_content, 
                    bulk_density, grain_size, foundation_depth, field_score
                )
                
                # Store in session state
                st.session_state['prediction'] = prediction
                st.session_state['soil_type'] = predicted_soil
                st.session_state['soil_data'] = soil_data
        
        # Display predictions if available
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            
            st.markdown("### 📊 AI PREDICTION RESULTS")
            
            col3, col4, col5, col6 = st.columns(4)
            
            with col3:
                st.markdown(f"""
                <div class='property-item'>
                    <div class='property-label'>Bearing Capacity</div>
                    <div class='property-value'>{pred['bearing_capacity']} kN/m²</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='property-item'>
                    <div class='property-label'>Safety Class</div>
                    <div class='property-value'>{pred['safety_class']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class='property-item'>
                    <div class='property-label'>Foundation Type</div>
                    <div class='property-value' style='font-size: 0.9rem;'>{soil_data['foundation_type'].split('(')[0]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                st.markdown(f"""
                <div class='property-item'>
                    <div class='property-label'>Recommended Depth</div>
                    <div class='property-value'>{soil_data['foundation_depth']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Safety probabilities
            st.markdown("### 🎯 Safety Classification Confidence")
            prob_cols = st.columns(len(pred['safety_probabilities']))
            for idx, (class_name, prob) in enumerate(pred['safety_probabilities'].items()):
                with prob_cols[idx]:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem; background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px;'>
                        <div style='color: #FFFFFF; font-weight: 600;'>{class_name}</div>
                        <div style='color: #00FF7F; font-size: 1.5rem; font-weight: 700;'>{prob}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(prob / 100)
            
            # Recommendations
            col7, col8 = st.columns(2)
            
            with col7:
                st.markdown("<div class='inspection-card'>", unsafe_allow_html=True)
                st.markdown("### 🏗️ RECOMMENDATIONS")
                for rec in soil_data['recommendations']:
                    st.markdown(f"<div class='recommendation-item'>✓ {rec}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col8:
                st.markdown("<div class='inspection-card'>", unsafe_allow_html=True)
                st.markdown("### ⚠️ WARNINGS")
                for warning in soil_data['warnings']:
                    if '✓' in warning:
                        st.markdown(f"<div class='success-box'>{warning}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='warning-box'>{warning}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Location map
            st.markdown("### 🗺️ SITE LOCATION")
            site_map = get_location_map(latitude, longitude, location_input)
            folium_static(site_map, width=1200, height=400)

with tab2:
    st.markdown("### 📊 Manual Soil Property Entry")
    
    if tabular_model:
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            manual_soil = st.selectbox("Soil Type", ["Sand", "Gravel", "Silt", "Clay"])
            manual_moisture = st.number_input("Moisture Content (%)", 0.0, 100.0, 25.0, 0.1)
        
        with col_m2:
            manual_density = st.number_input("Bulk Density (kN/m³)", 10.0, 25.0, 16.0, 0.1)
            manual_grain = st.selectbox("Grain Size", ["Fine", "Medium", "Coarse"], key="grain_manual")
        
        with col_m3:
            manual_depth = st.number_input("Foundation Depth (m)", 0.5, 5.0, 1.5, 0.1)
            manual_score = st.slider("Field Score", 1, 5, 3)
        
        if st.button("🔮 Predict Properties", type="primary"):
            manual_pred = predict_tabular(
                tabular_model, manual_soil, manual_moisture,
                manual_density, manual_grain, manual_depth, manual_score
            )
            
            st.success("✅ Prediction Complete!")
            
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.metric("Bearing Capacity", f"{manual_pred['bearing_capacity']} kN/m²")
            
            with col_r2:
                st.metric("Safety Class", manual_pred['safety_class'])
            
            st.markdown("### Safety Probabilities")
            for class_name, prob in manual_pred['safety_probabilities'].items():
                st.write(f"**{class_name}:** {prob}%")
                st.progress(prob / 100)
    else:
        st.error("❌ Tabular model not loaded. Please run tabular.py first.")

with tab3:
    st.markdown("### 📄 INSPECTION REPORT")
    
    if st.button("📥 Generate Report", type="primary"):
        report_data = {
            'Project': [project_name],
            'Engineer': [engineer_name],
            'Date': [test_date],
            'Location': [location_input],
            'Soil Type': [st.session_state.get('soil_type', 'N/A')],
            'Bearing Capacity': [f"{st.session_state.get('prediction', {}).get('bearing_capacity', 'N/A')} kN/m²"],
            'Safety Class': [st.session_state.get('prediction', {}).get('safety_class', 'N/A')],
        }
        
        df = pd.DataFrame(report_data)
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV Report",
            csv,
            f"inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

st.markdown("""
<div style='text-align: center; color: #666; margin-top: 4rem; padding: 2rem; border-top: 1px solid #222;'>
    <p style='font-size: 1.2rem; font-weight: 700; color: #FFF; text-transform: uppercase;'>
        ⚙️ POWERED BY DEEP LEARNING & MACHINE LEARNING
    </p>
    <p style='opacity: 0.7;'>ResNet50 + Random Forest • Real-time Analysis • GPS Integration</p>
</div>
""", unsafe_allow_html=True)