import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
import io
import os
import numpy as np
import time
from groq import Groq
import requests
import json
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

# Your Hugging Face model repo
MODEL_REPO = "eymenslimani/plant-disease-detector"

# Class labels (37 classes based on your code)
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

# Title and description
st.title("🌿 Plant Disease Detection")
st.write("Upload a photo of a plant leaf to detect if it's healthy or diseased.")

# Sidebar Debug Info
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
                    "downloads": model_info.get("downloads", 0)
                })
            elif response.status_code == 404:
                st.error("❌ Model NOT FOUND")
                st.info("👉 Visit: https://huggingface.co/" + model_name)
            elif response.status_code == 401:
                st.error("❌ Unauthorized")
            else:
                st.error(f"❌ Error: {response.status_code}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.info(f"**Current Model:** {MODEL_REPO}")
    
    if hf_token:
        st.success(f"🔑 HF Token: Set")
    else:
        st.warning("🔑 HF Token: Not set (may cause issues)")
    
    if groq_TOKEN:
        st.success("🔑 Groq Token: Set")

# Load model with better error handling
@st.cache_resource(show_spinner=False)
def load_model():
    """Load model with proper error handling for PyTorch 2.6+"""
    
    # Try different model files and configurations
    configs_to_try = [
        # (filename, load_function, model_architecture)
        ("best_model.pth", lambda p: torch.load(p, map_location='cpu', weights_only=False), 'tf_efficientnetv2_m.in21k_ft_in1k'),
        ("model.safetensors", lambda p: load_file(p), 'tf_efficientnetv2_m.in21k_ft_in1k'),
    ]
    
    for file_name, load_fn, arch in configs_to_try:
        try:
            st.info(f"🔄 Downloading {file_name}...")
            
            weights_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=file_name,
                token=hf_token,
                cache_dir="./model_cache"
            )
            
            st.success(f"✅ Downloaded {file_name}")
            st.info(f"📦 Loading weights...")
            
            # Load state dict
            state_dict = load_fn(weights_path)
            
            # Handle different state dict formats
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Create model
            st.info(f"🏗️ Creating model architecture...")
            model = timm.create_model(
                arch,
                pretrained=False,
                num_classes=NUM_CLASSES
            )
            
            # Load weights with strict=False to allow partial loading
            try:
                model.load_state_dict(state_dict, strict=True)
                st.success(f"✅ Model loaded successfully from {file_name}")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    st.warning(f"⚠️ Size mismatch detected, trying non-strict loading...")
                    model.load_state_dict(state_dict, strict=False)
                    st.success(f"✅ Model loaded (non-strict) from {file_name}")
                else:
                    raise
            
            model.eval()
            return model, file_name
            
        except Exception as e:
            error_msg = str(e)
            st.warning(f"❌ Failed to load {file_name}: {error_msg[:150]}...")
            continue
    
    # If all attempts fail, show detailed error
    st.error("❌ Could not load model from any available file")
    st.error("""
    **Possible Solutions:**
    
    1. **Re-export your model**: The model architecture doesn't match the saved weights.
       ```python
       # In your training script, save with:
       torch.save(model.state_dict(), 'best_model.pth')
       ```
    
    2. **Check NUM_CLASSES**: Make sure NUM_CLASSES matches your training (currently: {})
    
    3. **Manual fix**: Download the model and check the architecture used during training
    """.format(NUM_CLASSES))
    
    raise Exception("Failed to load model from HuggingFace")

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
    # Display uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("🔍 Analyzing image..."):
            try:
                # Load model
                model, model_file = load_model()
                processor = get_processor()
                
                st.success(f"✅ Model loaded from: {model_file}")
                
                # Process image
                img_array = np.array(image.convert("RGB"))
                augmented = processor(image=img_array)
                input_tensor = augmented['image'].unsqueeze(0)
                
                # Get predictions
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
                
                # Sort by confidence
                result = sorted(result, key=lambda x: x['score'], reverse=True)
                
                # Get top prediction
                top_prediction = result[0]
                label = top_prediction['label']
                confidence = top_prediction['score'] * 100
                
                # Display prediction
                st.success(f"**Prediction:** {label}")
                st.metric("Confidence", f"{confidence:.2f}%")
                
                # Show top 3 predictions
                with st.expander("📊 View all predictions"):
                    for i, pred in enumerate(result[:3], 1):
                        st.write(f"{i}. **{pred['label']}** - {pred['score']*100:.2f}%")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                
                with st.expander("🔍 See detailed error"):
                    import traceback
                    st.code(traceback.format_exc())
                
                st.info("""
                **Troubleshooting:**
                
                1. **500 Server Error** - HuggingFace servers are having issues. Try again in 15-30 minutes.
                2. **Model not found** - Verify model exists at: https://huggingface.co/eymenslimani/plant-disease-detector
                3. **Token issues** - Add HF_TOKEN to Streamlit secrets if model is private
                4. **Network timeout** - Check your internet connection
                
                **Alternative:** Download model files manually and deploy locally.
                """)
                st.stop()
    
    # Check if healthy
    is_healthy = "healthy" in label.lower()
    
    if is_healthy:
        st.success("✅ The plant appears healthy! No further action needed.")
    else:
        st.warning("⚠️ Disease detected. Get treatment advice below.")
        
        # Initialize chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "current_diagnosis" not in st.session_state or st.session_state.current_diagnosis != label:
            st.session_state.current_diagnosis = label
            st.session_state.messages = []
        
        # System prompt
        system_prompt = f"""You are a plant disease expert. The diagnosed disease is '{label}' with {confidence:.1f}% confidence.

Provide:
1. Brief explanation of the disease
2. Practical treatment solutions
3. Prevention tips
4. Answer follow-up questions

Be concise and use simple language."""
        
        # Display chat
        st.markdown("### 💬 Chat for Solutions")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("Ask about solutions or more details..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            messages = [
                {"role": "system", "content": system_prompt},
            ] + st.session_state.messages
            
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
st.markdown("💡 **Tip:** Upload clear, well-lit images of plant leaves for best results.")
