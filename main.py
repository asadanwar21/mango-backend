from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
import tempfile
from typing import List
import logging

# === Logger Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === App Setup ===
app = FastAPI()

# === Allow frontend to access backend ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Models ===
try:
    yolo_model = YOLO("runs/detect/mango_vs_nonmango/weights/best.pt")
    cnn_model = load_model("best_mango_disease_model.keras")
    label_map = joblib.load("label_encoder.pkl")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise Exception(f"Model loading failed: {str(e)}")

IMAGE_SIZE = 320

# === Preprocessing for CNN ===
def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Failed to read image: {img_path}")
            return None, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))

        norm = gray_resized.astype("float32") / 255.0
        norm = np.expand_dims(norm, axis=-1)  # (H, W, 1)
        norm = np.expand_dims(norm, axis=0)   # (1, H, W, 1)

        return norm, gray_resized
    except Exception as e:
        logger.error(f"Error preprocessing image {img_path}: {str(e)}")
        return None, None

# === Estimate severity if anthracnose ===
def estimate_anthracnose_severity(gray_img):
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_img)
        _, thresh = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY_INV)

        spot_area = np.sum(thresh == 255)
        total_area = gray_img.shape[0] * gray_img.shape[1]
        severity = (spot_area / total_area) * 100
        return round(severity, 2)
    except Exception as e:
        logger.error(f"Error estimating anthracnose severity: {str(e)}")
        return 0.0

# === Main Prediction Route ===
@app.get("/")
def read_root():
    return {"message": "Mango Disease Detection API is running!"}

@app.post("/analyze/")
async def analyze_images(files: List[UploadFile] = File(...)):
    try:
        if len(files) != 4:
            logger.error(f"Received {len(files)} files, expected 4")
            raise HTTPException(status_code=400, detail="Exactly 4 images are required.")

        individual_results = []
        temp_files = []
        mango_images = []

        for i, file in enumerate(files):
            logger.info(f"Processing file {i+1}: {file.filename}")
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(await file.read())
                img_path = tmp.name
                temp_files.append(img_path)

            # YOLO detection
            results = yolo_model(img_path, conf=0.662, verbose=False)[0]
            status = ""
            prediction_text = ""
            disease = ""
            confidence = 0

            if results.boxes is not None and len(results.boxes.cls) > 0:
                classes = [int(cls) for cls in results.boxes.cls]
                if 0 in classes and 1 not in classes:  # Mango detected
                    image_array, gray_img = preprocess_image(img_path)
                    if image_array is None:
                        logger.error(f"Image read failed for {file.filename}")
                        prediction_text = "🚫 Either not a mango or not properly captured"
                        status = "invalid"
                        disease = "N/A"
                        confidence = "N/A"
                    else:
                        preds = cnn_model.predict(image_array, verbose=0)[0]
                        pred_index = np.argmax(preds)
                        pred_class = label_map.inverse_transform([pred_index])[0]
                        confidence = float(preds[pred_index]) * 100

                        disease = pred_class
                        if pred_class == "healthy":
                            prediction_text = f"🟢 Healthy mango"
                        elif pred_class == "sap-burn":
                            prediction_text = f"🟡 Healthy mango (with sap burn)"
                        elif pred_class == "anthracnose":
                            severity = estimate_anthracnose_severity(gray_img)
                            prediction_text = f"🔴 Anthracnose detected – severity: {severity}%"
                        else:
                            prediction_text = f"❓ Unknown class"

                        status = "mango"
                        mango_images.append({
                            "image_number": f"Image {i + 1}",
                            "prediction": prediction_text,
                            "confidence": f"{confidence:.2f}%",
                            "disease": disease
                        })
                elif 1 in classes and 0 not in classes:
                    status = "not_mango"
                    prediction_text = "🚫 Either not a mango or not properly captured"
                    disease = "N/A"
                    confidence = "N/A"
                elif 0 in classes and 1 in classes:
                    status = "mixed"
                    prediction_text = "🚫 Either not a mango or not properly captured"
                    disease = "N/A"
                    confidence = "N/A"
                else:
                    status = "unknown"
                    prediction_text = "🚫 Either not a mango or not properly captured"
                    disease = "N/A"
                    confidence = "N/A"
            else:
                status = "no_object"
                prediction_text = "🚫 Either not a mango or not properly captured"
                disease = "N/A"
                confidence = "N/A"

            individual_results.append({
                "image_number": f"Image {i + 1}",
                "prediction": prediction_text,
                "confidence": confidence,
                "disease": disease,
                "status": status
            })

        # Check if at least 3 images are mangoes
        mango_count = sum(1 for res in individual_results if res["status"] == "mango")
        if mango_count < 3:
            invalid_images = [
                res["image_number"] for res in individual_results if res["status"] != "mango"
            ]
            error_message = (
               
                f" Invalid images: {', '.join(invalid_images)} are either not mangoes or not properly captured."
            )
            logger.info(error_message)
            return {
                "status": "error",
                "message": error_message,
                "individual_results": individual_results
            }

        # Aggregate results for final result based on mango images
        diseases = [res["disease"] for res in mango_images if res["disease"] != "N/A"]
        confidences = [
            float(res["confidence"].replace("%", "")) for res in mango_images
            if res["confidence"] != "N/A"
        ]

        # Majority voting for final disease
        if diseases:
            disease_counts = {}
            for disease in diseases:
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
            final_disease = max(disease_counts, key=disease_counts.get)
        else:
            final_disease = "unknown"

        # Average confidence
        final_confidence = (
            f"{sum(confidences) / len(confidences):.2f}%" if confidences else "N/A"
        )

        # Final prediction based on majority disease
        final_prediction = next(
            (res["prediction"] for res in mango_images if res["disease"] == final_disease),
            "Unknown"
        )

        return {
            "status": "success",
            "individual_results": individual_results,
            "final_result": {
                "prediction": final_prediction,
                "confidence": final_confidence,
                "disease": final_disease,
            },
        }

    except Exception as e:
        logger.error(f"Error in /analyze/: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))  # fallback to 8000 for local use
    uvicorn.run(app, host="0.0.0.0", port=port)


