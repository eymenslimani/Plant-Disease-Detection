import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
import io
import os
import numpy as np
from groq import Groq
import requests
import json
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from safetensors.torch import load_file

# Set up Hugging Face token if needed (for private repos, but yours is public)
hf_token = st.secrets.get("HF_TOKEN", None)

# Set up Groq client with your API key
groq_TOKEN = st.secrets["GROQ_API_KEY"]
groq_client = Groq(api_key=groq_TOKEN)

# Your Hugging Face model repo
MODEL_REPO = "eymenslimani/plant-disease-detector"

# Class labels from your model card (28 classes)
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
st.write("Upload a photo of a plant leaf to detect if it's healthy or diseased. If diseased, chat below for solutions and advice.")

# Add model status checker in sidebar
with st.sidebar:
    st.header("🔧 Debug Info")
    
    # Model name input for easy correction
    model_name = st.text_input("Model Repository", value=MODEL_REPO)
    
    if st.button("Check Model Status"):
        try:
            API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
            headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
            response = requests.get(API_URL, headers=headers)
            
            st.write(f"**Testing:** `{model_name}`")
            st.write(f"**Status Code:** {response.status_code}")
            
            if response.status_code == 200:
                st.success("✅ Model is accessible!")
                st.json(response.json()[:500] if len(str(response.json())) > 500 else response.json())
            elif response.status_code == 404:
                st.error("❌ Model NOT FOUND")
                st.warning("**Possible reasons:**\n1. Model name is incorrect\n2. Model is private and token doesn't have access\n3. Model hasn't been uploaded yet")
                st.info("👉 Try visiting: https://huggingface.co/" + model_name)
            elif response.status_code == 401:
                st.error("❌ Unauthorized - Token issue")
                st.info("Check your HF_TOKEN in secrets")
            else:
                st.error(f"❌ Error: {response.status_code}")
                with st.expander("See response"):
                    st.code(response.text[:1000])
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.info(f"**Current Model:** {MODEL_REPO}")
    
    # Token status
    if hf_token:
        st.success(f"🔑 HF Token: Set ({hf_token[:8]}...)")
    else:
        st.warning("🔑 HF Token: Not set")
    
    st.success("🔑 Groq Token: Set")

# Image upload
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run inference
    with st.spinner("🔍 Analyzing image..."):
        try:
            # Load model with caching and fallback
            @st.cache_resource
            def load_model():
                try:
                    # First try safetensors
                    weights_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.safetensors", token=hf_token)
                    state_dict = load_file(weights_path)
                except Exception as e:
                    st.warning(f"Failed to load model.safetensors: {str(e)[:100]}")
                    st.info("Falling back to best_model.pth...")
                    weights_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.pth", token=hf_token)
                    state_dict = torch.load(weights_path)
                
                model = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=False, num_classes=NUM_CLASSES)
                model.load_state_dict(state_dict)
                model.eval()  # Set to evaluation mode
                return model

            # Preprocessing transforms
            @st.cache_resource
            def get_processor():
                return A.Compose([
                    A.Resize(height=256, width=256),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])

            with st.spinner("Loading model from HuggingFace... (this may take a minute on first load)"):
                model = load_model()
                processor = get_processor()

            # Process image
            img_array = np.array(image.convert("RGB"))
            augmented = processor(image=img_array)
            input_tensor = augmented['image'].unsqueeze(0)  # Add batch dim

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
            
            st.success("✅ Model loaded successfully using timm and PyTorch")

            # Sort results by confidence score
            result = sorted(result, key=lambda x: x['score'], reverse=True)
            
            # Get top prediction
            top_prediction = result[0]
            label = top_prediction['label']
            confidence = top_prediction['score'] * 100

            # Display prediction with confidence
            st.success(f"**Prediction:** {label}")
            st.metric("Confidence", f"{confidence:.2f}%")
            
            # Show top 3 predictions
            with st.expander("View all predictions"):
                for i, pred in enumerate(result[:3], 1):
                    st.write(f"{i}. **{pred['label']}** - {pred['score']*100:.2f}%")

            # Check if healthy
            is_healthy = "healthy" in label.lower()

            if is_healthy:
                st.info("✅ The plant appears healthy! No further action needed.")
            else:
                st.warning("⚠️ Disease detected. Chat below for solutions and advice.")

                # Initialize chat session if not exists
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                
                # Store current diagnosis in session state
                if "current_diagnosis" not in st.session_state or st.session_state.current_diagnosis != label:
                    st.session_state.current_diagnosis = label
                    st.session_state.messages = []  # Reset chat for new diagnosis

                # System prompt for LLM
                system_prompt = f"""You are a plant disease expert assistant. The diagnosed disease is '{label}' with {confidence:.1f}% confidence.

Provide:
1. Brief explanation of the disease
2. Practical treatment solutions
3. Prevention tips for the future
4. Answer any follow-up questions

Be helpful, concise, and use simple language that farmers and gardeners can understand."""

                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # User input
                if prompt := st.chat_input("Ask about solutions or more details..."):
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Generate response with history
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
                                
                                # Add assistant response to history
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            except Exception as e:
                                st.error(f"Error generating response: {str(e)}")
                                st.info("Please try asking again.")

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            
            # Show detailed error for debugging
            with st.expander("🔍 See detailed error"):
                import traceback
                st.code(traceback.format_exc())
            
            st.info("**Possible issues:**\n\n1. **Model is still loading** - Wait 5-10 minutes after uploading\n2. **Image format issue** - Try JPG\n3. **Dependencies missing** - Check requirements.txt\n4. **Network issues** - Refresh\n\n💡 Test model at https://huggingface.co/eymenslimani/plant-disease-detector")

# Add footer with info
st.markdown("---")
st.markdown("💡 **Tip:** For best results, upload clear, well-lit images of plant leaves.")
