import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Cats vs Dogs Classifier", page_icon="🐾")

st.title("🐾 Cats vs Dogs Image Classifier")
st.write("FastAPI Backend aur Streamlit Frontend se juda Model Integration.")

# File Uploader Widget
uploaded_file = st.file_uploader("Image Choose Karein...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Class"):
        with st.spinner("FastAPI Server Se Predict Ho Raha Hai..."):
            try:
                # Image ko bytes mein convert karke FastAPI endpoint par bhejna
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                img_bytes = img_byte_arr.getvalue()

                files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
                
                # FastAPI Local Backend Request (POST /predict)
                response = requests.post("http://127.0.0.1:8000/predict", files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"**Prediction:** {result['prediction']}")
                    st.metric(label="Probability Score", value=f"{result['probability']}")
                else:
                    st.error("Backend se prediction prapt nahi ho saki.")

            except Exception as e:
                st.error(f"FastAPI Server Connect Nahi Ho Paya. Pehle Backend Start Karein! Error: {e}")