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
    .main {
        background: #f8f9fa;
    }
    
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: white;
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f1f3f5;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: #495057;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    .soil-type {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 1rem 0;
        text-align: center;
    }
    
    .chat-box {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .user-msg {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.2);
    }
    
    .bot-msg {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        box-shadow: 0 2px 8px rgba(156, 39, 176, 0.2);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .quick-btn {
        background: white;
        border: 2px solid #667eea;
        color: #667eea;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 0.3rem;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .quick-btn:hover {
        background: #667eea;
        color: white;
    }
    
    .info-tip {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🌱 AI Soil Analyzer</h1><p>Smart Classification • Live Camera • AI Assistant</p></div>', unsafe_allow_html=True)

# Session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load dataset
@st.cache_data
def load_dataset_auto():
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
                arr = np.array(img.resize((100, 100))).astype(float)
                dataset.append({
                    "soil": soil_type,
                    "array": arr,
                    "filename": os.path.basename(img_path)
                })
            except Exception:
                continue
    
    return dataset

dataset = load_dataset_auto()

# Classification functions
def calculate_blur(img):
    gray = np.array(img.convert("L"))
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
    variance = 0
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            patch = gray[i-1:i+2, j-1:j+2]
            conv = np.sum(patch * laplacian)
            variance += conv ** 2
    
    variance = variance / (gray.shape[0] * gray.shape[1])
    return variance

def extract_features(img):
    arr = np.array(img.resize((100, 100))).astype(float)
    gray = np.array(img.convert("L").resize((100, 100))).astype(float)
    return arr, gray

def similarity_score(arr1, arr2):
    return np.mean((arr1 - arr2) ** 2)

def classify_soil(img):
    blur_score = calculate_blur(img)
    
    # Check image quality but don't reject completely
    quality_penalty = 0
    quality_note = ""
    
    if blur_score < 30:
        return None, 0, "Image too blurry - unable to analyze", True
    elif blur_score < 80:
        quality_penalty = 20
        quality_note = " (Poor Quality)"
    elif blur_score > 5000:
        quality_penalty = 15
        quality_note = " (Noisy Image)"
    
    img_arr, gray = extract_features(img)
    
    if len(dataset) > 0:
        best_match = None
        best_score = float('inf')
        
        for item in dataset:
            score = similarity_score(img_arr, item["array"])
            if score < best_score:
                best_score = score
                best_match = item
        
        if best_score < 500:
            return best_match["soil"], max(98 - quality_penalty, 60), f"AI Match{quality_note}", False
        elif best_score < 1500:
            return best_match["soil"], max(92 - quality_penalty, 55), f"Strong Match{quality_note}", False
        elif best_score < 3000:
            return best_match["soil"], max(85 - quality_penalty, 50), f"Good Match{quality_note}", False
    
    mean_gray = gray.mean()
    std_gray = gray.std()
    
    if std_gray < 20:
        return "clay", max(72 - quality_penalty, 45), f"AI Prediction{quality_note}", False
    elif mean_gray > 180:
        return "sand", max(75 - quality_penalty, 45), f"AI Prediction{quality_note}", False
    elif mean_gray > 140:
        return "silt", max(70 - quality_penalty, 45), f"AI Prediction{quality_note}", False
    else:
        return "gravel", max(68 - quality_penalty, 45), f"AI Prediction{quality_note}", False

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
        return "**Clay Soil:**\n• Very fine particles\n• High cohesion\n• Low drainage\n• Settles over time\n• Needs deep foundations"
    
    elif 'sand' in q:
        return "**Sandy Soil:**\n• Coarse particles\n• Good drainage\n• High bearing capacity\n• Immediate settlement\n• Best for shallow foundations"
    
    elif 'spt' in q:
        return "**SPT Test** = Measures soil strength\n\n**N-values:**\n• N < 10: Loose/Soft\n• N = 10-30: Medium\n• N > 30: Dense/Hard\n\nHigher N = Stronger soil!"
    
    elif 'shear' in q:
        return "**Shear Strength** = Soil's resistance to sliding\n\n**Formula:** τ = c + σ tan(φ)\n\nClay has high c, Sand has high φ"
    
    elif 'hello' in q or 'hi' in q:
        return "👋 Hi! I'm your Soil Mechanics AI.\n\nAsk me about:\n• Bearing capacity\n• Foundations\n• Soil types (clay/sand)\n• Settlement\n• SPT tests"
    
    else:
        return "🤖 **I can help with:**\n\n• Bearing capacity\n• Foundation types\n• Settlement analysis\n• Soil properties\n• SPT testing\n• Shear strength\n\nJust ask a question!"

# TABS
tab1, tab2, tab3 = st.tabs(["📸 Upload Image", "📹 Live Camera", "🤖 AI Assistant"])

# TAB 1 - Upload
with tab1:
    uploaded_file = st.file_uploader("Drop soil image here", type=["jpg", "png", "jpeg", "jfif"], label_visibility="collapsed")
    
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
                    time.sleep(0.003)
                    progress.progress(i + 1)
                
                soil_type, confidence, method, is_error = classify_soil(img)
                
                if is_error:
                    st.error(f"❌ {method}")
                else:
                    st.markdown(f'<div class="result-card"><div class="soil-type">🎯 {soil_type.upper()}</div></div>', unsafe_allow_html=True)
                    st.progress(confidence / 100)
                    st.metric("Confidence", f"{confidence}%")
                    
                    if confidence >= 90:
                        st.success(f"✅ {method}")
                    else:
                        st.info(f"ℹ️ {method}")
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
                    st.button("🔄 Retake")
                else:
                    st.markdown(f'<div class="result-card"><div class="soil-type">🎯 {soil_type.upper()}</div></div>', unsafe_allow_html=True)
                    st.progress(confidence / 100)
                    st.metric("Accuracy", f"{confidence}%")
                    
                    if confidence >= 90:
                        st.success(f"✅ {method}")
                        st.balloons()
                    else:
                        st.info(f"ℹ️ {method}")
    else:
        st.info("👆 Click camera button to capture")

# TAB 3 - Chatbot
with tab3:
    st.markdown("### 💬 Ask About Soil Mechanics")
    
    # Chat display
    chat_container = st.container(height=400)
    
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown('<div class="info-tip">👋 <b>Welcome!</b> Ask me about soil mechanics, foundations, or soil types.</div>', unsafe_allow_html=True)
        
        for chat in st.session_state.chat_history:
            if chat['role'] == 'user':
                st.markdown(f'<div class="user-msg">👤 {chat["msg"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">🤖 {chat["msg"]}</div>', unsafe_allow_html=True)
    
    # Input
    user_input = st.chat_input("Type your question...")
    
    if user_input:
        st.session_state.chat_history.append({'role': 'user', 'msg': user_input})
        response = get_ai_response(user_input)
        st.session_state.chat_history.append({'role': 'bot', 'msg': response})
        st.rerun()
    
    # Quick questions
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
st.markdown("<p style='text-align: center; color: #999;'>🌱 Powered by AI & Computer Vision</p>", unsafe_allow_html=True)