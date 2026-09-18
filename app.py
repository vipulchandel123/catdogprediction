from fastapi import FastAPI, UploadFile, File, HTTPException
from model_loader import predict_image

app = FastAPI(title="Cats vs Dogs Classification API")

@app.get("/")
def home():
    return {"message": "Cats vs Dogs Classification API is Running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate content type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Sirf image files upload karein."
        )

    try:
        image_bytes = await file.read()
        probability, label, confidence = predict_image(image_bytes)

        return {
            "filename": file.filename,
            "prediction": label,
            "confidence": f"{confidence:.2f}%",
            "probability": probability,
            "class_index": 1 if label == "Dog" else 0
        }

    except Exception as e:
        print("Internal Processing Error:", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )