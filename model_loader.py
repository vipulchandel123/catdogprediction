import io
import numpy as np
from PIL import Image
import tensorflow as tf

# Model load logic
try:
    model = tf.keras.models.load_model("cat_dog_cnn_model.keras")
    print("Full Keras Model successfully loaded!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict_image(image_bytes: bytes):
    if model is None:
        raise Exception("Model load nahi hua hai. File check karein.")

    # Palette with transparency handling (Fixes PIL UserWarning)
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Correct input target size expected by Dense Layer (128x128)
    img = img.resize((128, 128))

    # Preprocessing
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    probability = float(predictions[0][0])

    if probability >= 0.5:
        label = "Dog"
        confidence = probability * 100
    else:
        label = "Cat"
        confidence = (1 - probability) * 100

    return round(probability, 4), label, round(confidence, 2)