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

# Class labels (37 classes)
LABELS = [
    "Apple_Scab_Leaf", "Apple_cedar_apple_rust_leaf", "Apple_healthy_leaf",
    "Bell_pepper_Bacterial_spot_leaf", "Bell_pepper_healthy_leaf",
    "Blueberry_healthy_leaf", "Cherry_Powdery_mildew_leaf", "Cherry_healthy_leaf",
    "Corn_Common_rust_leaf", "Corn_Gray_leaf_spot_leaf", "Corn_Northern_Leaf_Blight_leaf",
    "Corn_healthy_leaf", "Grape_Black_Measles_leaf", "Grape_Black_rot_leaf",
    "Grape_Leaf_blight_leaf", "Grape_healthy_leaf", "Peach_Bacterial_spot_leaf",
    "Peach_healthy_leaf", "Potato_Early_blight_leaf", "Potato_Late_blight_leaf",
    "Potato_healthy_leaf", "Raspberry_healthy_leaf", "Soybean_healthy_leaf",
    "Squash_Powdery_mildew_leaf", "Strawberry_Leaf_scorch_leaf", "Strawberry_healthy_leaf",
    "Tomato_Early_blight_leaf", "Tomato_Late_blight_leaf", "Tomato_Leaf_Mold_leaf",
    "Tomato_Septoria_leaf_spot_leaf", "Tomato_Spider_mites_Two_spotted_spider_mite_leaf",
    "Tomato_Target_Spot_leaf", "Tomato_Tomato_YellowLeaf_Curl_Virus_leaf",
    "Tomato_Tomato_mosaic_virus_leaf", "Tomato_bacterial_spot_leaf", "Tomato_healthy_leaf",
    "grape_leaf_black_rot"
]
NUM_CLASSES = len(LABELS)
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

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
    """Load model with proper PyTorch 2.6+ handling"""
    
    configs_to_try = [
        ("best_model.pth", lambda p: torch.load(p, map_location='cpu', weights_only=False)),
        ("model.safetensors", lambda p: load_file(p)),
    ]
    
    for file_name, load_fn in configs_to_try:
        try:
            st.info(f"🔄 Downloading {file_name}...")
            
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
            
            # Create model
            st.info("🏗️ Creating model...")
            model = timm.create_model(
                'tf_efficientnetv2_m.in21k_ft_in1k',
                pretrained=False,
                num_classes=NUM_CLASSES
            )
            
            # Try strict loading first
            try:
                model.load_state_dict(state_dict, strict=True)
                st.success(f"✅ Model loaded from {file_name}")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    st.warning("⚠️ Size mismatch, trying non-strict loading...")
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    st.success(f"✅ Model loaded (non-strict) from {file_name}")
                    if missing_keys:
                        st.warning(f"Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        st.warning(f"Unexpected keys: {len(unexpected_keys)}")
                else:
                    raise
            
            model.eval()
            return model, file_name
            
        except Exception as e:
            st.warning(f"❌ {file_name} failed: {str(e)[:100]}")
            continue
    
    st.error("❌ Could not load model")
    st.error(f"**Check:** Make sure NUM_CLASSES={NUM_CLASSES} matches your training setup")
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
                model, model_file = load_model()
                processor = get_processor()
                
                # Process image
                img_array = np.array(image.convert("RGB"))
                augmented = processor(image=img_array)
                input_tensor = augmented['image'].unsqueeze(0)
                
                # Predict
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probs, k=min(3, NUM_CLASSES))
                
                result = []
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    result.append({
                        'label': ID2LABEL[idx.item()],
                        'score': prob.item()
                    })
                
                result = sorted(result, key=lambda x: x['score'], reverse=True)
                
                # Display prediction
                top_prediction = result[0]
                label = top_prediction['label']
                confidence = top_prediction['score'] * 100
                
                st.success(f"**Prediction:** {label}")
                st.metric("Confidence", f"{confidence:.2f}%")
                
                with st.expander("📊 View all predictions"):
                    for i, pred in enumerate(result[:3], 1):
                        st.write(f"{i}. **{pred['label']}** - {pred['score']*100:.2f}%")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔍 Details"):
                    import traceback
                    st.code(traceback.format_exc())
                st.stop()
    
    # Check if healthy
    is_healthy = "healthy" in label.lower()
    
    if is_healthy:
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
