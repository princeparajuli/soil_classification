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
    .chat-user {
        background: #e3f2fd;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .chat-bot {
        background: #f3e5f5;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .welcome-msg {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('''
<div class="main-header">
    <h1>🌱 AI Soil Analyzer</h1>
    <p>Smart Classification • Live Camera • AI Assistant</p>
</div>
''', unsafe_allow_html=True)

# Session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Detect if image is actually soil or not
def detect_non_soil(img):
    """Detect if image contains text, uniform colors, or non-natural patterns"""
    arr = np.array(img.resize((128, 128))).astype(float)
    gray = np.array(img.convert("L").resize((128, 128))).astype(float)
    
    # Check 1: Too uniform (like white paper/notebook)
    std_gray = gray.std()
    if std_gray < 8:  # Very low variation = paper/uniform background
        return True, "uniform background (paper/notebook?)"
    
    # Check 2: Very high contrast (like text on paper)
    edges_h = np.abs(gray[1:, :] - gray[:-1, :])
    edges_v = np.abs(gray[:, 1:] - gray[:, :-1])
    edge_variance = (edges_h.std() + edges_v.std()) / 2
    
    if edge_variance > 35:  # Sharp edges = likely text/drawings
        return True, "text or drawings detected"
    
    # Check 3: Unnatural colors (too bright/saturated - like notebook covers)
    mean_rgb = np.mean(arr, axis=(0, 1))
    if np.all(mean_rgb > 200):  # Too white
        return True, "not a soil image"
    
    # Check 4: RGB channels too similar (grayscale document)
    rgb_std = np.std(mean_rgb)
    if rgb_std < 3 and std_gray > 20:  # Gray but with patterns = printed material
        return True, "grayscale document detected"
    
    return False, ""

# Enhanced feature extraction with soil-specific features
def extract_soil_features(img):
    """Extract features specifically for soil classification"""
    arr = np.array(img.resize((128, 128))).astype(float)
    
    # Color features
    mean_rgb = np.mean(arr, axis=(0, 1))
    std_rgb = np.std(arr, axis=(0, 1))
    
    # Brown/earthy tone detection
    r, g, b = mean_rgb
    brown_score = (r - b) / (r + g + b + 1)  # Brown has more red than blue
    
    # Convert to HSV for better color analysis
    arr_uint = arr.astype(np.uint8)
    img_hsv = Image.fromarray(arr_uint).convert('HSV')
    hsv_arr = np.array(img_hsv).astype(float)
    mean_hsv = np.mean(hsv_arr, axis=(0, 1))
    std_hsv = np.std(hsv_arr, axis=(0, 1))
    
    # Grayscale features
    gray = np.array(img.convert("L").resize((128, 128))).astype(float)
    mean_gray = gray.mean()
    std_gray = gray.std()
    
    # Texture features - CRITICAL for distinguishing soil types
    # Calculate local standard deviation (roughness)
    roughness = 0
    for i in range(0, 128-8, 8):
        for j in range(0, 128-8, 8):
            patch = gray[i:i+8, j:j+8]
            roughness += patch.std()
    roughness = roughness / ((128//8) * (128//8))
    
    # Edge detection for texture
    edges_h = np.abs(gray[1:, :] - gray[:-1, :]).mean()
    edges_v = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    edge_intensity = (edges_h + edges_v) / 2
    
    # Color variance (important for particle detection)
    color_variance = np.mean([std_rgb[0], std_rgb[1], std_rgb[2]])
    
    # Saturation (clay is less saturated, sand/gravel more)
    saturation = mean_hsv[1]
    
    return {
        'array': arr,
        'mean_rgb': mean_rgb,
        'std_rgb': std_rgb,
        'mean_hsv': mean_hsv,
        'std_hsv': std_hsv,
        'mean_gray': mean_gray,
        'std_gray': std_gray,
        'roughness': roughness,
        'edge_intensity': edge_intensity,
        'color_variance': color_variance,
        'saturation': saturation,
        'brown_score': brown_score
    }

# Data augmentation
def augment_image(img):
    """Create augmented versions of images"""
    augmented = [img]
    
    # Rotations
    augmented.append(img.rotate(10))
    augmented.append(img.rotate(-10))
    
    # Brightness adjustments
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(img)
    augmented.append(enhancer.enhance(0.85))
    augmented.append(enhancer.enhance(1.15))
    
    # Contrast
    enhancer = ImageEnhance.Contrast(img)
    augmented.append(enhancer.enhance(0.9))
    augmented.append(enhancer.enhance(1.1))
    
    return augmented

# Load dataset with augmentation
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
                
                # Add original and augmented versions
                augmented_imgs = augment_image(img)
                
                for aug_img in augmented_imgs:
                    features = extract_soil_features(aug_img)
                    dataset.append({
                        "soil": soil_type,
                        "features": features,
                        "filename": os.path.basename(img_path)
                    })
            except Exception:
                continue
    
    return dataset

dataset = load_dataset_enhanced()

# Soil-specific similarity with better weights
def calculate_soil_similarity(features1, features2):
    """Calculate similarity with emphasis on texture and color"""
    
    # Texture is KEY for distinguishing soil types
    roughness_diff = (features1['roughness'] - features2['roughness']) ** 2
    edge_diff = (features1['edge_intensity'] - features2['edge_intensity']) ** 2
    
    # Color features
    rgb_diff = np.mean((features1['mean_rgb'] - features2['mean_rgb']) ** 2)
    hsv_diff = np.mean((features1['mean_hsv'] - features2['mean_hsv']) ** 2)
    
    # Grayscale (brightness matters)
    gray_diff = (features1['mean_gray'] - features2['mean_gray']) ** 2
    
    # Color variance (particle size indicator)
    variance_diff = (features1['color_variance'] - features2['color_variance']) ** 2
    
    # Saturation
    sat_diff = (features1['saturation'] - features2['saturation']) ** 2
    
    # Weighted combination - TEXTURE IS MOST IMPORTANT
    total_score = (
        roughness_diff * 0.25 +      # Texture roughness
        edge_diff * 0.20 +            # Edge intensity
        variance_diff * 0.20 +        # Color variance
        hsv_diff * 0.15 +             # HSV color
        rgb_diff * 0.10 +             # RGB color
        gray_diff * 0.05 +            # Brightness
        sat_diff * 0.05               # Saturation
    )
    
    return total_score

# Enhanced classification with soil-specific rules
def classify_soil_enhanced(img):
    """Improved classification with better soil type distinction"""
    
    # First check if it's actually soil
    is_non_soil, reason = detect_non_soil(img)
    if is_non_soil:
        return "invalid", 0, f"Not a soil image - {reason}", True
    
    img_features = extract_soil_features(img)
    
    if len(dataset) == 0:
        return "unknown", 0, "No dataset available", True
    
    # Find best matches
    matches = []
    for item in dataset:
        score = calculate_soil_similarity(img_features, item['features'])
        matches.append({
            'soil': item['soil'],
            'score': score
        })
    
    # Sort by score
    matches.sort(key=lambda x: x['score'])
    
    # Vote from top 20 matches with weighted voting
    top_matches = matches[:20]
    vote_score = {}
    
    for i, match in enumerate(top_matches):
        soil = match['soil']
        # Closer matches get more weight
        weight = 20 - i
        vote_score[soil] = vote_score.get(soil, 0) + weight
    
    # Get winner
    predicted_soil = max(vote_score, key=vote_score.get)
    best_score = matches[0]['score']
    total_votes = vote_score[predicted_soil]
    max_possible_votes = sum(range(20, 0, -1))
    vote_percentage = (total_votes / max_possible_votes) * 100
    
    # Additional rule-based refinement for clay/silt/sand
    mean_gray = img_features['mean_gray']
    roughness = img_features['roughness']
    color_var = img_features['color_variance']
    
    # Clay: smooth, fine, uniform, darker
    # Silt: medium texture, medium brightness
    # Sand: coarse, grainy, lighter, varied
    # Gravel: very coarse, rocky, high variance
    
    # Apply soil-specific rules if confidence is borderline
    if predicted_soil in ['clay', 'silt', 'sand']:
        # Clay characteristics
        if roughness < 15 and color_var < 20 and mean_gray < 120:
            if predicted_soil != 'clay':
                # Override if very clay-like
                if roughness < 12:
                    predicted_soil = 'clay'
        
        # Sand characteristics
        elif roughness > 20 and color_var > 25 and mean_gray > 140:
            if predicted_soil != 'sand':
                if roughness > 25:
                    predicted_soil = 'sand'
        
        # Silt is in between
        elif 15 <= roughness <= 20 and 120 <= mean_gray <= 140:
            if predicted_soil not in ['silt']:
                predicted_soil = 'silt'
    
    # Calculate confidence
    if best_score < 30 and vote_percentage > 60:
        confidence = 94
        method = "Excellent Match"
    elif best_score < 60 and vote_percentage > 50:
        confidence = 87
        method = "Strong Match"
    elif best_score < 100 and vote_percentage > 40:
        confidence = 78
        method = "Good Match"
    elif best_score < 150 and vote_percentage > 30:
        confidence = 68
        method = "Probable Match"
    elif best_score < 250:
        confidence = 58
        method = "Moderate Confidence"
    else:
        confidence = 45
        method = "Low Confidence"
    
    return predicted_soil, confidence, method, False

# Simple chatbot
def get_ai_response(question):
    q = question.lower()
    
    if 'bearing' in q or 'capacity' in q:
        return "**Bearing Capacity** = Maximum load soil can support\n\n**Formula:** Qult = cNc + γDfNq + 0.5γBNγ\n\n**Safe capacity** = Qult ÷ 3 (with safety factor)"
    elif 'settlement' in q:
        return "**Settlement** = Foundation sinking into soil\n\n**3 Types:**\n• Immediate (instant)\n• Primary (time-based)\n• Secondary (long-term)\n\nClay settles more than sand!"
    elif 'foundation' in q or 'footing' in q:
        return "**Foundation Types:**\n\n**Shallow:** Isolated, Strip, Mat\n**Deep:** Piles, Drilled Shafts\n\nUse shallow if soil is strong near surface!"
    elif 'clay' in q:
        return "**Clay Soil:**\n• Very fine particles\n• Smooth texture\n• High cohesion\n• Low drainage\n• Dark color\n• Settles over time"
    elif 'sand' in q:
        return "**Sandy Soil:**\n• Coarse particles\n• Grainy texture\n• Good drainage\n• High bearing capacity\n• Light color\n• Immediate settlement"
    elif 'silt' in q:
        return "**Silt Soil:**\n• Medium particles\n• Smooth when wet\n• Medium drainage\n• Medium brown color\n• Between clay and sand"
    elif 'gravel' in q:
        return "**Gravel:**\n• Large particles/rocks\n• Very coarse\n• Excellent drainage\n• High bearing capacity\n• Rocky appearance"
    elif 'spt' in q:
        return "**SPT Test** = Measures soil strength\n\n**N-values:**\n• N < 10: Loose/Soft\n• N = 10-30: Medium\n• N > 30: Dense/Hard\n\nHigher N = Stronger soil!"
    elif 'shear' in q:
        return "**Shear Strength** = Soil's resistance to sliding\n\n**Formula:** τ = c + σ tan(φ)\n\nClay has high c, Sand has high φ"
    elif 'hello' in q or 'hi' in q:
        return "👋 Hi! I'm your Soil Mechanics AI.\n\nAsk me about:\n• Bearing capacity\n• Foundations\n• Soil types (clay/sand/silt)\n• Settlement\n• SPT tests"
    else:
        return "🤖 **I can help with:**\n\n• Bearing capacity\n• Foundation types\n• Settlement analysis\n• Soil properties\n• SPT testing\n• Shear strength\n\nJust ask a question!"

# TABS
tab1, tab2, tab3 = st.tabs(["📸 Upload Image", "📹 Live Camera", "🤖 AI Assistant"])

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
                st.info(f"📊 Dataset: {len(dataset)} augmented images")
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.003)
                    progress.progress(i + 1)
                
                soil_type, confidence, method, is_error = classify_soil_enhanced(img)
                
                if is_error:
                    st.error(f"❌ {method}")
                    st.warning("Please upload a clear soil sample image")
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
                        st.warning(f"⚠️ {method} - Consider retaking with better lighting")
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
                st.info(f"📊 Dataset: {len(dataset)} augmented images")
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.002)
                    progress.progress(i + 1)
                
                soil_type, confidence, method, is_error = classify_soil_enhanced(img)
                
                if is_error:
                    st.error(f"❌ {method}")
                    st.warning("Please capture a clear soil sample image")
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
                        st.warning(f"⚠️ {method} - Try better angle/lighting")
    else:
        st.info("👆 Click camera button to capture")

# TAB 3 - Chatbot
with tab3:
    st.markdown("### 💬 Ask About Soil Mechanics")
    
    chat_container = st.container(height=400)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown('''<div class="welcome-msg">
                👋 Welcome! Ask me about soil mechanics, foundations, or soil types.
            </div>''', unsafe_allow_html=True)
        
        for chat in st.session_state.chat_history:
            if chat['role'] == 'user':
                st.markdown(f'<div class="chat-user">👤 {chat["msg"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">🤖 {chat["msg"]}</div>', 
                           unsafe_allow_html=True)
    
    user_input = st.chat_input("Type your question...")
    if user_input:
        st.session_state.chat_history.append({'role': 'user', 'msg': user_input})
        response = get_ai_response(user_input)
        st.session_state.chat_history.append({'role': 'bot', 'msg': response})
        st.rerun()
    
    st.markdown("**⚡ Quick Ask:**")
    col1, col2, col3, col4 = st.columns(4)
    questions = [
        "Bearing capacity?",
        "Foundation types?",
        "Clay vs Sand?",
        "SPT test?"
    ]
    cols = [col1, col2, col3, col4]
    
    for i, q in enumerate(questions):
        with cols[i]:
            if st.button(q, key=f"q{i}", use_container_width=True):
                st.session_state.chat_history.append({'role': 'user', 'msg': q})
                response = get_ai_response(q)
                st.session_state.chat_history.append({'role': 'bot', 'msg': response})
                st.rerun()

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #666;">🌱 Powered by AI & Computer Vision | Upload SOIL images only</div>', 
           unsafe_allow_html=True)