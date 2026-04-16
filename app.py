import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
import gdown
import os

st.set_page_config(page_title="Land Cover Classification", page_icon="🛰️")
st.title("🌍 Land Cover Classification")
st.markdown("Upload a satellite image to classify land cover types")

CLASS_NAMES = ['Urban', 'Agriculture', 'Forest', 'Water']
COLORS = [(128,64,128), (34,139,34), (0,128,0), (70,130,180)]

# Google Drive file ID (from your link)
FILE_ID = "1IfnnZjZrDQFdbWB-3R3rvhZyJV2WJBRp"
MODEL_PATH = "best_model.pth"

@st.cache_resource
def load_model():
    """Download model from Google Drive if not exists, then load it"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner(" Downloading model file (25MB)... This may take a minute"):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            st.success(" Model downloaded!")
    
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=4
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
    return output.argmax(dim=1).squeeze(0).cpu().numpy()

def colorize(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(COLORS):
        colored[mask == i] = color
    return colored

# Load model
try:
    model = load_model()
    st.success(" Model ready!")
    model_ok = True
except Exception as e:
    st.error(f"❌ Error: {e}")
    model_ok = False

if model_ok:
    uploaded = st.file_uploader("Choose a satellite image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded:
        original = Image.open(uploaded).convert('RGB')
        
        col1, col2 = st.columns(2)
        col1.image(original, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Classifying..."):
            tensor = preprocess(original)
            mask = predict(model, tensor)
            colored = colorize(mask)
            result = Image.fromarray(colored)
            result = result.resize(original.size)
        
        col2.image(result, caption="Classification Result", use_container_width=True)
        
        st.subheader(" Land Cover Distribution")
        total = mask.size
        for i, name in enumerate(CLASS_NAMES):
            pct = (mask == i).sum() / total * 100
            st.write(f"{name}: {pct:.1f}%")
        
        import io
        buf = io.BytesIO()
        result.save(buf, format='PNG')
        st.download_button(" Download Result", data=buf.getvalue(), file_name="result.png")
