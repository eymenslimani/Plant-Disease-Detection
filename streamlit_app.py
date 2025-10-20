import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import time
from groq import Groq
import requests
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from safetensors.torch import load_file

# Set page config
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

# Set up tokens
hf_token = st.secrets.get("HF_TOKEN", None)
groq_TOKEN = st.secrets.get("GROQ_API_KEY", None)

if groq_TOKEN:
    groq_client = Groq(api_key=groq_TOKEN)
else:
    st.error("❌ GROQ_API_KEY not found in secrets!")
    st.stop()

# Model repository
MODEL_REPO = "eymenslimani/plant-disease-detector"

# PlantDoc Dataset - 27 classes
LABELS_27 = [
    "Apple_leaf",
    "Apple_rust_leaf",
    "Bell_pepper_leaf",
    "Blueberry_leaf",
    "Cherry_leaf",
    "Corn_Gray_leaf_spot",
    "Corn_leaf_blight",
    "Peach_leaf",
    "Potato_leaf_early_blight",
    "Potato_leaf_late_blight",
    "Raspberry_leaf",
    "Soyabean_leaf",
    "Soybean_leaf",
    "Squash_Powdery_mildew_leaf",
    "Strawberry_leaf",
    "Tomato_Early_blight_leaf",
    "Tomato_Septoria_leaf_spot",
    "Tomato_leaf",
    "Tomato_leaf_bacterial_spot",
    "Tomato_leaf_late_blight",
    "Tomato_leaf_mosaic_virus",
    "Tomato_leaf_yellow_virus",
    "Tomato_mold_leaf",
    "Tomato_two_spotted_spider_mites_leaf",
    "grape_leaf",
    "grape_leaf_black_rot",
    "Corn_rust_leaf"
]

# Alternative 38 classes (PlantVillage style)
LABELS_38 = [
    "Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy",
    "Blueberry_healthy", "Cherry_(including_sour)_Powdery_mildew", "Cherry_(including_sour)_healthy",
    "Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot", "Corn_(maize)_Common_rust_",
    "Corn_(maize)_Northern_Leaf_Blight", "Corn_(maize)_healthy",
    "Grape_Black_rot", "Grape_Esca_(Black_Measles)", "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape_healthy",
    "Orange_Haunglongbing_(Citrus_greening)", "Peach_Bacterial_spot", "Peach_healthy",
    "Pepper,_bell_Bacterial_spot", "Pepper,_bell_healthy",
    "Potato_Early_blight", "Potato_Late_blight", "Potato_healthy",
    "Raspberry_healthy", "Soybean_healthy",
    "Squash_Powdery_mildew", "Strawberry_Leaf_scorch", "Strawberry_healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two-spotted_spider_mite",
    "Tomato_Target_Spot", "Tomato_Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

# Title
st.title("🌿 Plant Disease Detection")
st.write("Upload a photo of a plant leaf to detect if it's healthy or diseased.")

# Sidebar
with st.sidebar:
    st.header("🔧 Debug Info")
    
    model_name = st.text_input("Model Repository", value=MODEL_REPO)
    
    if st.button("Check Model Status"):
        try:
            API_URL = f"https://huggingface.co/api/models/{model_name}"
            headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
            response = requests.get(API_URL, headers=headers, timeout=10)
            
            st.write(f"**Testing:** `{model_name}`")
            st.write(f"**Status Code:** {response.status_code}")
            
            if response.status_code == 200:
                st.success("✅ Model is accessible!")
                model_info = response.json()
                st.json({
                    "id": model_info.get("id"),
                    "author": model_info.get("author"),
                    "lastModified": model_info.get("lastModified"),
                })
            else:
                st.error(f"❌ Error: {response.status_code}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.info(f"**Model:** {MODEL_REPO}")
    
    if hf_token:
        st.success("🔑 HF Token: Set")
    else:
        st.warning("🔑 HF Token: Not set")
    
    if groq_TOKEN:
        st.success("🔑 Groq Token: Set")

# Load model
@st.cache_resource(show_spinner=False)
def load_model():
    """Load model with auto-detection of num_classes"""
    
    # Try different file formats and class counts
    configs = [
        ("model.safetensors", load_file),
        ("best_model.pth", lambda p: torch.load(p, map_location='cpu', weights_only=False)),
        ("pytorch_model.bin", lambda p: torch.load(p, map_location='cpu', weights_only=False)),
    ]
    
    num_classes_to_try = [27, 38, 30, 37, 39]  # Common numbers
    
    for file_name, load_fn in configs:
        try:
            st.info(f"🔄 Trying {file_name}...")
            
            weights_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=file_name,
                token=hf_token,
                cache_dir="./model_cache"
            )
            
            st.success(f"✅ Downloaded {file_name}")
            
            # Load state dict
            state_dict = load_fn(weights_path)
            
            # Handle nested dicts
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
            
            # Try to detect num_classes from the state dict
            detected_classes = None
            for key in state_dict.keys():
                if 'classifier' in key and 'weight' in key:
                    weight_shape = state_dict[key].shape
                    if len(weight_shape) >= 1:
                        detected_classes = weight_shape[0]
                        st.info(f"🔍 Detected {detected_classes} classes from model weights")
                        break
            
            # Add detected classes to the list to try first
            if detected_classes and detected_classes not in num_classes_to_try:
                num_classes_to_try.insert(0, detected_classes)
            
            # Try different numbers of classes
            for num_classes in num_classes_to_try:
                try:
                    st.info(f"🏗️ Trying with {num_classes} classes...")
                    
                    model = timm.create_model(
                        'tf_efficientnetv2_m.in21k_ft_in1k',
                        pretrained=False,
                        num_classes=num_classes
                    )
                    
                    # Try loading
                    model.load_state_dict(state_dict, strict=True)
                    st.success(f"✅ Success! Model has {num_classes} classes")
                    
                    model.eval()
                    
                    # Select appropriate labels
                    if num_classes == 27:
                        labels = LABELS_27
                    elif num_classes == 38:
                        labels = LABELS_38
                    else:
                        labels = [f"Class_{i}" for i in range(num_classes)]
                    
                    return model, file_name, num_classes, labels
                    
                except RuntimeError as e:
                    if "size mismatch" not in str(e):
                        st.warning(f"⚠️ {num_classes} classes: {str(e)[:80]}")
                    continue
            
            st.warning(f"❌ Could not match any class count for {file_name}")
            
        except Exception as e:
            st.warning(f"⚠️ {file_name}: {str(e)[:100]}")
            continue
    
    st.error("❌ Could not load model with any configuration")
    st.error("Try checking your model file or contact the model creator")
    raise Exception("Failed to load model")

# Preprocessing
@st.cache_resource
def get_processor():
    return A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# Image upload
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image")
    
    with col2:
        with st.spinner("🔍 Analyzing..."):
            try:
                # Load model
                model, model_file, num_classes, labels = load_model()
                processor = get_processor()
                
                st.info(f"📊 Using model with {num_classes} classes")
                
                # Process image
                img_array = np.array(image.convert("RGB"))
                augmented = processor(image=img_array)
                input_tensor = augmented['image'].unsqueeze(0)
                
                # Predict
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probs, k=min(5, num_classes))
                
                result = []
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    idx_val = idx.item()
                    label = labels[idx_val] if idx_val < len(labels) else f"Class_{idx_val}"
                    result.append({
                        'label': label,
                        'score': prob.item()
                    })
                
                # Display prediction
                top_prediction = result[0]
                label = top_prediction['label']
                confidence = top_prediction['score'] * 100
                
                st.success(f"**Prediction:** {label}")
                st.metric("Confidence", f"{confidence:.2f}%")
                
                with st.expander("📊 Top 5 predictions"):
                    for i, pred in enumerate(result[:5], 1):
                        st.write(f"{i}. **{pred['label']}** - {pred['score']*100:.2f}%")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔍 Details"):
                    import traceback
                    st.code(traceback.format_exc())
                st.stop()
    
    # Check if healthy or disease
    is_disease = any(keyword in label.lower() for keyword in 
                    ['blight', 'rust', 'spot', 'rot', 'virus', 'mildew', 'mites', 'mold', 'scab', 'scorch'])
    
    if not is_disease and 'healthy' not in label.lower():
        st.info("ℹ️ Classification complete. Check the disease keywords for details.")
    elif 'healthy' in label.lower():
        st.success("✅ The plant appears healthy!")
    else:
        st.warning("⚠️ Disease detected. Get advice below.")
        
        # Initialize chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "current_diagnosis" not in st.session_state or st.session_state.current_diagnosis != label:
            st.session_state.current_diagnosis = label
            st.session_state.messages = []
        
        # System prompt
        system_prompt = f"""You are a plant disease expert. Diagnosed: '{label}' ({confidence:.1f}% confidence).

Provide:
1. Disease explanation
2. Treatment solutions
3. Prevention tips
4. Answer questions

Be concise and clear."""
        
        # Chat interface
        st.markdown("### 💬 Ask for Solutions")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about treatment..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chat_completion = groq_client.chat.completions.create(
                            messages=messages,
                            model="llama3-8b-8192",
                            temperature=0.7,
                            max_tokens=800,
                        )
                        response = chat_completion.choices[0].message.content
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Use clear, well-lit images for best results.")
