import streamlit as st
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("📝 Handwritten OCR with TrOCR")

def check_model_cache():
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    model_name = "microsoft/trocr-base-printed"
    model_path = Path(cache_dir) / "hub" / model_name.replace("/", "--")
    return model_path.exists()

# Initialize model and processor only once
@st.cache_resource
def load_model():
    try:
        model_name = "microsoft/trocr-base-printed"
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        
        # Check if model is already cached
        if check_model_cache():
            st.info("🔄 Loading model from local cache... (This is fast!)")
        else:
            st.info("⏳ Downloading model for the first time... (This might take a few minutes)")
        
        processor = TrOCRProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False,
            use_fast=True
        )
        
        model = VisionEncoderDecoderModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Show device info
        if device.type == "cuda":
            st.success(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ Using CPU (this will be slower)")
        
        model.to(device)
        model.eval()
        
        # Test model with a simple input
        with st.spinner("Testing model..."):
            test_input = torch.randn(1, 3, 384, 384).to(device)
            with torch.no_grad():
                test_output = model.generate(test_input, max_length=20)
        
        st.success("✅ Model loaded and ready!")
        return processor, model, device
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        return None, None, None

# Show loading status
with st.spinner("Initializing..."):
    processor, model, device = load_model()

if processor is None or model is None:
    st.error("Model failed to load. Please check the logs and try again.")
    st.stop()

uploaded_file = st.file_uploader("Upload a handwritten image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Display image info for debugging
        st.write(f"Image size: {image.size}, Mode: {image.mode}")
        
        with st.spinner("Recognizing text..."):
            # Process image
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            logger.info(f"Processed image shape: {pixel_values.shape}")
            logger.info(f"Pixel values range: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")
            
            # Move to device
            pixel_values = pixel_values.to(device)
            
            # Generate text with more specific parameters
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    max_length=128,
                    num_beams=5,
                    length_penalty=1.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            # Decode prediction
            predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.info(f"Generated IDs shape: {generated_ids.shape}")
            logger.info(f"Predicted text: {predicted_text}")
            
            st.success("✅ Text recognized!")
            st.text_area("📜 Predicted Text", predicted_text, height=150)
            
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        st.error(f"An error occurred during text recognition: {str(e)}")
