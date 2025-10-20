import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import json
from groq import Groq
import requests
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from safetensors.torch import load_file

# Set page config
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

# Set up tokens with user input fallback
hf_token = st.secrets.get("HF_TOKEN", None)
groq_token = st.secrets.get("GROQ_API_KEY", None)

if not groq_token:
    groq_token = st.sidebar.text_input("Enter GROQ_API_KEY", type="password")
    if not groq_token:
        st.sidebar.error("❌ GROQ_API_KEY not set! Please enter a valid API key.")
        st.stop()

try:
    groq_client = Groq(api_key=groq_token)
except Exception as e:
    st.sidebar.error(f"❌ Invalid GROQ_API_KEY: {str(e)}")
    st.stop()

# Model repository
MODEL_REPO = "eymenslimani/plant-disease-detector"

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
    
    if groq_token:
        st.success("🔑 Groq Token: Set")

# Load model and labels
@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    """Load model and try to get labels from config"""
    
    # Try to load label mapping from HF repo
    label_files_to_try = ["config.json", "label_map.json", "labels.json", "id2label.json"]
    labels = None
    
    for label_file in label_files_to_try:
        try:
            config_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=label_file,
                token=hf_token,
                cache_dir="./model_cache"
            )
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            if 'id2label' in config_data:
                labels = {int(k): v for k, v in config_data['id2label'].items()}
                st.success(f"✅ Loaded labels from {label_file}")
                break
            elif 'labels' in config_data:
                labels = {i: label for i, label in enumerate(config_data['labels'])}
                st.success(f"✅ Loaded labels from {label_file}")
                break
        except:
            continue
    
    # Try different file formats
    configs = [
        ("model.safetensors", load_file),
        ("best_model.pth", lambda p: torch.load(p, map_location='cpu', weights_only=False)),
        ("pytorch_model.bin", lambda p: torch.load(p, map_location='cpu', weights_only=False)),
    ]
    
    for file_name, load_fn in configs:
        try:
            st.info(f"🔄 Loading {file_name}...")
            
            weights_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=file_name,
                token=hf_token,
                cache_dir="./model_cache"
            )
            
            state_dict = load_fn(weights_path)
            
            # Handle nested dicts
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
            
            # Detect num_classes from state dict
            detected_classes = None
            for key in state_dict.keys():
                if 'classifier' in key and 'weight' in key:
                    weight_shape = state_dict[key].shape
                    if len(weight_shape) >= 1:
                        detected_classes = weight_shape[0]
                        st.info(f"🔍 Detected {detected_classes} classes")
                        break
            
            if detected_classes is None:
                continue
            
            # Create model
            st.info(f"🏗️ Creating model with {detected_classes} classes...")
            model = timm.create_model(
                'tf_efficientnetv2_m.in21k_ft_in1k',
                pretrained=False,
                num_classes=detected_classes
            )
            
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            
            # Use loaded labels or create generic ones
            if labels is None:
                labels = {i: f"Disease_Class_{i}" for i in range(detected_classes)}
                st.warning(f"⚠️ Using generic labels. Upload label_map.json to your model repo for proper names.")
            
            st.success(f"✅ Model loaded successfully!")
            return model, file_name, detected_classes, labels
            
        except Exception as e:
            st.warning(f"⚠️ {file_name}: {str(e)[:100]}")
            continue
    
    st.error("❌ Could not load model")
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
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("🔍 Analyzing..."):
            try:
                # Load model
                model, model_file, num_classes, labels = load_model_and_labels()
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
                top_probs, top_indices = torch.topk(probs, k=min(5, num_classes))
                
                result = []
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    idx_val = idx.item()
                    label = labels.get(idx_val, f"Class_{idx_val}")
                    result.append({
                        'label': label,
                        'score': prob.item(),
                        'index': idx_val
                    })
                
                # Display prediction
                top_prediction = result[0]
                label = top_prediction['label']
                confidence = top_prediction['score'] * 100
                
                # Better formatting for the prediction
                st.markdown("### 🎯 Diagnosis Result")
                st.success(f"**{label}**")
                st.metric("Confidence Level", f"{confidence:.1f}%")
                
                with st.expander("📊 View Top 5 Predictions"):
                    for i, pred in enumerate(result[:5], 1):
                        confidence_bar = "█" * int(pred['score'] * 20)
                        st.write(f"{i}. **{pred['label']}**")
                        st.progress(pred['score'])
                        st.caption(f"{pred['score']*100:.2f}%")
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                with st.expander("🔍 Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
                st.stop()
    
    # Always show chat interface
    st.markdown("---")
    st.markdown("## 💬 Ask Plant Disease Expert")
    
    # Check if it's a disease or healthy
    is_healthy = 'healthy' in label.lower()
    
    if is_healthy:
        st.success("✅ The plant appears healthy! You can still ask questions below.")
        initial_context = f"This plant appears to be healthy ({label}). "
    else:
        st.warning("⚠️ Disease or issue detected. Get expert advice below.")
        initial_context = f"Detected condition: {label} with {confidence:.1f}% confidence. "
    
    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_diagnosis" not in st.session_state or st.session_state.current_diagnosis != label:
        st.session_state.current_diagnosis = label
        st.session_state.messages = []
    
    # System prompt
    system_prompt = f"""You are an expert plant pathologist and agricultural advisor. 

Current diagnosis: '{label}' (Confidence: {confidence:.1f}%)

Your role:
1. Explain what this diagnosis means in clear, simple terms
2. Provide specific treatment recommendations
3. Suggest prevention strategies
4. Answer any follow-up questions about plant care
5. If the label is generic (like "Class_X"), acknowledge this and provide general plant disease advice

Be practical, concise, and farmer-friendly in your responses."""
    
    # Show chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Auto-generate first response if no messages
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing diagnosis..."):
                try:
                    initial_prompt = f"Based on the diagnosis of '{label}', please provide: 1) What this means, 2) Treatment recommendations, 3) Prevention tips. Keep it concise and practical."
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": initial_prompt}
                    ]
                    
                    chat_completion = groq_client.chat.completions.create(
                        messages=messages,
                        model="llama-3.3-70b-versatile",
                        temperature=0.7,
                        max_tokens=1000,
                    )
                    response = chat_completion.choices[0].message.content
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating advice: {str(e)}")
    
    # Chat input
    if prompt := st.chat_input("Ask about treatment, prevention, or care..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chat_completion = groq_client.chat.completions.create(
                        messages=messages,
                        model="llama-3.3-70b-versatile",
                        temperature=0.7,
                        max_tokens=1000,
                    )
                    response = chat_completion.choices[0].message.content
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** For best results, use clear, well-lit photos showing the affected leaf area.")
