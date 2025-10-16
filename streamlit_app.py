import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
import os
from groq import Groq

# Set up Hugging Face Inference Client (no token needed for public models)
hf_client = InferenceClient()

# Set up Groq client with your API key (replace with your new key after regeneration)
GROQ_API_KEY = "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"  # Temporary; regenerate and replace
groq_client = Groq(api_key=GROQ_API_KEY)

# Placeholder for your Hugging Face model repo
MODEL_REPO = "eymenslimani/plant-disease-detector"  # Replace with your actual model, e.g., "yourusername/your-plant-model"

# Title and description
st.title("Plant Disease Detection")
st.write("Upload a photo of a plant leaf to detect if it's healthy or diseased. If diseased, chat below for solutions and advice.")

# Image upload
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to bytes for inference
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Run inference
    with st.spinner("Analyzing image..."):
        try:
            result = hf_client.image_classification(img_bytes, model=MODEL_REPO)
            # Assume result is a list of dicts with 'score' and 'label'
            top_prediction = max(result, key=lambda x: x['score'])
            label = top_prediction['label']
            confidence = top_prediction['score'] * 100

            st.success(f"Prediction: {label} (Confidence: {confidence:.2f}%)")

            # Check if healthy (adjust based on your model's labels; common pattern is 'healthy' in label)
            is_healthy = "healthy" in label.lower()

            if is_healthy:
                st.info("The plant appears healthy! No further action needed.")
            else:
                st.warning("Disease detected. Chat below for solutions and advice.")

                # Initialize chat session if not exists
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # System prompt for LLM, primed with diagnosis
                system_prompt = f"You are a plant disease expert. The diagnosed disease is '{label}'. Provide practical solutions, treatments, prevention tips, and answer any follow-up questions based on this diagnosis. Be helpful, concise, and use simple language."

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
                            chat_completion = groq_client.chat.completions.create(
                                messages=messages,
                                model="llama3-8b-8192",  # Free, decent model
                                temperature=0.7,
                                max_tokens=512,
                            )
                            response = chat_completion.choices[0].message.content
                            st.markdown(response)

                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})

       
