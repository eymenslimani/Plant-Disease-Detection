import streamlit as st
from huggingface_hub import InferenceClient, hf_hub_download
from PIL import Image
import io
import os
from groq import Groq
import requests
import json
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Set up Hugging Face Inference Client with token
hf_token = st.secrets.get("HF_TOKEN", None)  # Get HF token if it exists
hf_client = InferenceClient(token=hf_token)

# Set up Groq client with your API key
groq_TOKEN = st.secrets["GROQ_API_KEY"]
groq_client = Groq(api_key=groq_TOKEN)

# Your Hugging Face model repo
MODEL_REPO = "eymenslimani/plant-disease-detector"

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

    # Convert to bytes for inference
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Run inference
    with st.spinner("🔍 Analyzing image..."):
        try:
            # Method 1: Try loading model directly with transformers
            @st.cache_resource
            def load_model():
                try:
                    processor = AutoImageProcessor.from_pretrained(MODEL_REPO, token=hf_token, trust_remote_code=True)
                    model = AutoModelForImageClassification.from_pretrained(
                        MODEL_REPO, 
                        token=hf_token,
                        trust_remote_code=True
                    )
                    return processor, model
                except Exception as e:
                    st.error(f"Model loading error: {str(e)}")
                    raise
            
            try:
                with st.spinner("Loading model from HuggingFace... (this may take a minute on first load)"):
                    processor, model = load_model()
                
                # Process image
                inputs = processor(images=image, return_tensors="pt")
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                # Get top predictions
                top_probs, top_indices = torch.topk(probs, k=min(3, len(model.config.id2label)))
                
                result = []
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    result.append({
                        'label': model.config.id2label[idx.item()],
                        'score': prob.item()
                    })
                
                st.success("✅ Using direct model loading")
                
            except Exception as e1:
                error_msg = str(e1)
                st.warning(f"Direct loading failed: {error_msg[:200]}...")
                
                # Show detailed error
                with st.expander("🔍 See full error"):
                    st.code(error_msg)
                
                st.info("Trying InferenceClient...")
                
                # Method 2: Try InferenceClient
                img_bytes.seek(0)
                try:
                    result = hf_client.image_classification(
                        img_bytes, 
                        model=MODEL_REPO,
                    )
                except Exception as e2:
                    st.warning(f"InferenceClient failed: {str(e2)[:100]}")
                    st.info("Trying direct API call...")
                    
                    # Method 3: Direct API call as fallback
                    img_bytes.seek(0)
                    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_REPO}"
                    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
                    
                    response = requests.post(API_URL, headers=headers, data=img_bytes.read())
                    
                    if response.status_code == 503:
                        raise Exception("Model is loading on HuggingFace servers. Please wait 1-2 minutes and try again.")
                    elif response.status_code != 200:
                        raise Exception(f"API returned status code {response.status_code}: {response.text[:200]}")
                    
                    result = response.json()
            
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

            # Check if healthy (adjust based on your model's labels)
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

                # System prompt for LLM, primed with diagnosis
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
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())
            
            st.info("**Possible issues:**\n\n1. **Model is still loading** - Wait 5-10 minutes after uploading your model\n2. **Image format issue** - Try a different image or convert to JPG\n3. **Model not public** - Check if your model is set to public on Hugging Face\n4. **Network issues** - Refresh and try again\n\n💡 **Quick fix:** Visit your model page at https://huggingface.co/eymenslimani/plant-disease-detector and try the Inference API widget there first.")

# Add footer with info
st.markdown("---")
st.markdown("💡 **Tip:** For best results, upload clear, well-lit images of plant leaves.")
