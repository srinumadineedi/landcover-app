import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

st.set_page_config(page_title="Land Cover Classification", page_icon="🛰️")
st.title("🌍 Land Cover Classification")
st.markdown("Upload a satellite image to classify land cover types")

CLASS_NAMES = ['Urban', 'Agriculture', 'Forest', 'Water']
COLORS = [(128,64,128), (34,139,34), (0,128,0), (70,130,180)]

@st.cache_resource
def load_model():
    return ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])

def preprocess(image):
    image = image.resize((64, 64))
    img = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def predict(model, input_tensor):
    outputs = model.run(['output'], {'input': input_tensor})[0]
    mask = np.argmax(outputs[0], axis=0)
    return mask

def colorize(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(COLORS):
        colored[mask == i] = color
    return colored

try:
    session = load_model()
    st.success("✅ Model loaded successfully")
    model_ok = True
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model_ok = False

if model_ok:
    uploaded = st.file_uploader("Choose a satellite image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded:
        original = Image.open(uploaded).convert('RGB')
        
        col1, col2 = st.columns(2)
        col1.image(original, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Classifying..."):
            input_tensor = preprocess(original)
            mask = predict(session, input_tensor)
            colored = colorize(mask)
            result = Image.fromarray(colored)
            result = result.resize(original.size)
        
        col2.image(result, caption="Classification Result", use_container_width=True)
        
        st.subheader("Land Cover Distribution")
        total = mask.size
        for i, name in enumerate(CLASS_NAMES):
            pct = (mask == i).sum() / total * 100
            st.write(f"{name}: {pct:.1f}%")
        
        import io
        buf = io.BytesIO()
        result.save(buf, format='PNG')
        st.download_button("📥 Download Result", data=buf.getvalue(), file_name="result.png")
