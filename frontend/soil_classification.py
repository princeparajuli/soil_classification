import streamlit as st
from PIL import Image
import numpy as np
import os
import glob
import time

st.set_page_config(page_title="AI Soil Analysis", layout="wide", page_icon="🌱")

# Modern CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .soil-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('''
<div class="main-header">
    <h1>🌱 AI Soil Analyzer</h1>
    <p>Smart Classification • Live Camera • Instant Results</p>
</div>
''', unsafe_allow_html=True)

# Detect non-soil images (notebooks, papers, screenshots)
def is_valid_soil_image(img):
    """Check if image is actually soil, not paper/notebook/screenshot"""
    gray = np.array(img.convert("L").resize((100, 100))).astype(float)
    arr = np.array(img.resize((100, 100))).astype(float)
    
    std_gray = gray.std()
    mean_gray = gray.mean()
    mean_rgb = np.mean(arr, axis=(0, 1))
    
    # Reject white paper/notebook
    if mean_gray > 200 and std_gray < 15:
        return False
    
    # Reject very uniform images (blank pages)
    if std_gray < 5:
        return False
    
    # Reject pure white/black screenshots
    if np.all(mean_rgb > 240) or np.all(mean_rgb < 15):
        return False
    
    # Reject images with text (very sharp edges)
    edges = np.abs(gray[1:, :] - gray[:-1, :])
    if edges.max() > 200 and edges.std() > 40:
        return False
    
    return True

# Extract comprehensive soil features
def extract_soil_features(img):
    """Extract features that distinguish soil types accurately"""
    arr = np.array(img.resize((150, 150))).astype(float)
    gray = np.array(img.convert("L").resize((150, 150))).astype(float)
    
    # Basic color features
    mean_rgb = np.mean(arr, axis=(0, 1))
    std_rgb = np.std(arr, axis=(0, 1))
    
    # Grayscale analysis
    mean_gray = gray.mean()
    std_gray = gray.std()
    
    # CRITICAL: Particle texture analysis
    # Measure local variations (particle size indicator)
    particle_texture = 0
    patch_count = 0
    for i in range(0, 140, 10):
        for j in range(0, 140, 10):
            patch = gray[i:i+10, j:j+10]
            particle_texture += patch.std()
            patch_count += 1
    particle_texture = particle_texture / patch_count
    
    # Edge analysis (roughness/graininess)
    edges_h = np.abs(gray[1:, :] - gray[:-1, :])
    edges_v = np.abs(gray[:, 1:] - gray[:, :-1])
    edge_mean = (edges_h.mean() + edges_v.mean()) / 2
    edge_std = (edges_h.std() + edges_v.std()) / 2
    
    # Color uniformity (clay is uniform, gravel is varied)
    color_uniformity = 1.0 / (std_gray + 1)
    
    # HSV for better color detection
    arr_uint = arr.astype(np.uint8)
    img_hsv = Image.fromarray(arr_uint).convert('HSV')
    hsv_arr = np.array(img_hsv).astype(float)
    mean_hsv = np.mean(hsv_arr, axis=(0, 1))
    
    # Brown/earth tone (soil specific)
    r, g, b = mean_rgb
    earth_tone = (r + g) / (b + 1) if b > 0 else 1
    
    return {
        'array': arr,
        'mean_rgb': mean_rgb,
        'std_rgb': std_rgb,
        'mean_gray': mean_gray,
        'std_gray': std_gray,
        'particle_texture': particle_texture,
        'edge_mean': edge_mean,
        'edge_std': edge_std,
        'color_uniformity': color_uniformity,
        'mean_hsv': mean_hsv,
        'earth_tone': earth_tone
    }

# Minimal augmentation (only 3 versions per image for speed)
def augment_image(img):
    """Light augmentation to avoid over-processing"""
    augmented = [img]
    
    from PIL import ImageEnhance
    # Slight brightness variation
    enhancer = ImageEnhance.Brightness(img)
    augmented.append(enhancer.enhance(0.9))
    augmented.append(enhancer.enhance(1.1))
    
    return augmented

# Load dataset efficiently
@st.cache_data
def load_dataset_enhanced():
    dataset = []
    possible_paths = ["dataset", "./dataset", "../dataset", "Dataset", "./Dataset"]
    dataset_path = None
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            dataset_path = path
            break
    
    if not dataset_path:
        return []
    
    soil_folders = [f for f in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, f))]
    
    for soil_folder in soil_folders:
        soil_type = soil_folder.lower()
        folder_path = os.path.join(dataset_path, soil_folder)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.jfif', '*.JPG', '*.JPEG', '*.PNG', '*.JFIF']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                augmented_imgs = augment_image(img)
                
                for aug_img in augmented_imgs:
                    features = extract_soil_features(aug_img)
                    dataset.append({
                        "soil": soil_type,
                        "features": features
                    })
            except Exception:
                continue
    
    return dataset

dataset = load_dataset_enhanced()

# Advanced similarity calculation
def calculate_similarity(f1, f2):
    """Smart similarity focusing on what matters for each soil type"""
    
    # Texture features (MOST IMPORTANT for gravel vs others)
    particle_diff = (f1['particle_texture'] - f2['particle_texture']) ** 2
    edge_mean_diff = (f1['edge_mean'] - f2['edge_mean']) ** 2
    edge_std_diff = (f1['edge_std'] - f2['edge_std']) ** 2
    
    # Color features (important for clay/silt/sand)
    gray_diff = (f1['mean_gray'] - f2['mean_gray']) ** 2
    std_diff = (f1['std_gray'] - f2['std_gray']) ** 2
    rgb_diff = np.mean((f1['mean_rgb'] - f2['mean_rgb']) ** 2)
    
    # Uniformity (clay is uniform, gravel is chaotic)
    uniformity_diff = (f1['color_uniformity'] - f2['color_uniformity']) ** 2
    
    # HSV color space
    hsv_diff = np.mean((f1['mean_hsv'] - f2['mean_hsv']) ** 2)
    
    # Weighted scoring - TEXTURE DOMINATES
    score = (
        particle_diff * 0.30 +      # Particle texture
        edge_mean_diff * 0.20 +     # Edge intensity
        edge_std_diff * 0.15 +      # Edge variation
        std_diff * 0.12 +           # Gray variation
        uniformity_diff * 0.10 +    # Uniformity
        rgb_diff * 0.08 +           # RGB color
        hsv_diff * 0.05             # HSV color
    )
    
    return score

# Smart classification with soil-specific logic
def classify_soil(img):
    """Final classification with strong gravel detection"""
    
    # Validate soil image first
    if not is_valid_soil_image(img):
        return None, 0, "Not a soil image - please upload actual soil sample", True
    
    if len(dataset) == 0:
        return None, 0, "No dataset loaded", True
    
    img_features = extract_soil_features(img)
    
    # Calculate all similarities
    matches = []
    for item in dataset:
        score = calculate_similarity(img_features, item['features'])
        matches.append({
            'soil': item['soil'],
            'score': score
        })
    
    # Sort by best match
    matches.sort(key=lambda x: x['score'])
    
    # Weighted voting from top 25 matches
    vote_weights = {}
    for i in range(min(25, len(matches))):
        soil = matches[i]['soil']
        weight = 25 - i  # Closer matches get more weight
        vote_weights[soil] = vote_weights.get(soil, 0) + weight
    
    # Get prediction
    predicted_soil = max(vote_weights, key=vote_weights.get)
    best_score = matches[0]['score']
    
    # CRITICAL: Apply soil-specific rules for accuracy
    pt = img_features['particle_texture']
    mg = img_features['mean_gray']
    sg = img_features['std_gray']
    em = img_features['edge_mean']
    
    # GRAVEL detection (most distinctive - large rocks/particles)
    if pt > 30 or em > 25 or sg > 45:
        # Strong gravel indicators
        gravel_votes = vote_weights.get('gravel', 0)
        if gravel_votes > 50 or (pt > 35 and em > 20):
            predicted_soil = 'gravel'
    
    # CLAY detection (smooth, uniform, fine)
    elif pt < 12 and sg < 20 and mg < 130:
        clay_votes = vote_weights.get('clay', 0)
        if clay_votes > 40 or (pt < 10 and sg < 15):
            predicted_soil = 'clay'
    
    # SAND detection (grainy, light colored)
    elif pt > 18 and mg > 140 and sg > 25:
        sand_votes = vote_weights.get('sand', 0)
        if sand_votes > 40 or (pt > 22 and mg > 150):
            predicted_soil = 'sand'
    
    # SILT detection (medium everything)
    elif 12 <= pt <= 18 and 100 <= mg <= 140 and 20 <= sg <= 30:
        silt_votes = vote_weights.get('silt', 0)
        if silt_votes > 40:
            predicted_soil = 'silt'
    
    # Calculate confidence
    max_votes = sum(range(1, 26))
    vote_percent = (vote_weights[predicted_soil] / max_votes) * 100
    
    if best_score < 20 and vote_percent > 60:
        confidence = 95
        method = "Excellent Match"
    elif best_score < 40 and vote_percent > 50:
        confidence = 88
        method = "Strong Match"
    elif best_score < 70 and vote_percent > 40:
        confidence = 78
        method = "Good Match"
    elif best_score < 120 and vote_percent > 30:
        confidence = 65
        method = "Probable"
    else:
        confidence = 52
        method = "Low Confidence"
    
    return predicted_soil, confidence, method, False

# TABS
tab1, tab2 = st.tabs(["📸 Upload Image", "📹 Live Camera"])

# TAB 1 - Upload
with tab1:
    uploaded_file = st.file_uploader("Drop soil image here", type=["jpg", "png", "jpeg", "jfif"], 
                                     label_visibility="collapsed")
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.image(img, use_container_width=True)
        
        with col2:
            if len(dataset) == 0:
                st.error("⚠️ No training data found")
            else:
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.002)
                    progress.progress(i + 1)
                
                soil_type, confidence, method, is_error = classify_soil(img)
                
                if is_error:
                    st.error(f"❌ {method}")
                else:
                    st.markdown(f'<div class="soil-result">🎯 {soil_type.upper()}</div>', 
                               unsafe_allow_html=True)
                    st.progress(confidence / 100)
                    st.metric("Confidence", f"{confidence}%")
                    
                    if confidence >= 80:
                        st.success(f"✅ {method}")
                    elif confidence >= 60:
                        st.info(f"ℹ️ {method}")
                    else:
                        st.warning(f"⚠️ {method}")
    else:
        st.info("👆 Upload a soil image to analyze")

# TAB 2 - Camera
with tab2:
    camera_photo = st.camera_input("📸 Capture soil sample")
    
    if camera_photo:
        img = Image.open(camera_photo)
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.image(img, use_container_width=True)
        
        with col2:
            if len(dataset) == 0:
                st.error("⚠️ No training data")
            else:
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.002)
                    progress.progress(i + 1)
                
                soil_type, confidence, method, is_error = classify_soil(img)
                
                if is_error:
                    st.error(f"❌ {method}")
                else:
                    st.markdown(f'<div class="soil-result">🎯 {soil_type.upper()}</div>', 
                               unsafe_allow_html=True)
                    st.progress(confidence / 100)
                    st.metric("Accuracy", f"{confidence}%")
                    
                    if confidence >= 80:
                        st.success(f"✅ {method}")
                        st.balloons()
                    elif confidence >= 60:
                        st.info(f"ℹ️ {method}")
                    else:
                        st.warning(f"⚠️ {method}")
    else:
        st.info("👆 Click camera button to capture")

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #666;">🌱 AI Soil Classification System</div>', 
           unsafe_allow_html=True)