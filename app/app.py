from pathlib import Path
import sys
import shutil
import uuid

import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# =========================================================
# 1. PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


# =========================================================
# 2. LANGGRAPH
# =========================================================

from src.assistant_graph import assistant_graph


# =========================================================
# 3. FASTAPI APP
# =========================================================

app = FastAPI(
    title="Brain CT Stroke AI Assistant"
)


# =========================================================
# 4. PATHS
# =========================================================

APP_DIR = PROJECT_ROOT / "app"

UPLOAD_DIR = APP_DIR / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INDEX_FILE = APP_DIR / "index.html"

STYLE_FILE = APP_DIR / "style.css"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "brain_ct_resnet18_weighted.pth"
)


# =========================================================
# 5. STATIC FILES
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory=str(APP_DIR)),
    name="static"
)


# =========================================================
# 6. CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "Normal",
    "Ischemia",
    "Bleeding"
]


# =========================================================
# 7. DEVICE
# =========================================================

device = torch.device("cpu")


# =========================================================
# 8. LOAD MODEL
# =========================================================

print("========================================")
print("Loading Brain CT model...")
print("========================================")


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )


model = models.resnet18(
    weights=None
)


model.fc = nn.Linear(
    model.fc.in_features,
    3
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model.to(device)

model.eval()


print("Brain CT model loaded successfully.")
print("Model:", MODEL_PATH)
print("Device:", device)
print("========================================")


# =========================================================
# 9. IMAGE PREPROCESSING
#
# MUST MATCH TRAINING
#
# Resize -> ToTensor
# No Normalize
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

])


# =========================================================
# 10. HOME PAGE
# =========================================================

@app.get("/")
async def home():

    if not INDEX_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail="index.html not found."
        )

    return FileResponse(
        INDEX_FILE
    )


# =========================================================
# 11. HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():

    return {
        "status": "online",
        "model": "Brain CT ResNet18",
        "device": "cpu"
    }


# =========================================================
# 12. PREDICTION
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Check filename
    # -----------------------------------------------------

    original_filename = (
        file.filename
        or "uploaded_image.png"
    )


    # -----------------------------------------------------
    # Check file type
    # -----------------------------------------------------

    allowed_types = [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "image/bmp"
    ]


    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid image format. "
                "Please upload PNG, JPG, JPEG, WEBP "
                "or BMP image."
            )
        )


    # -----------------------------------------------------
    # Generate unique filename
    # -----------------------------------------------------

    extension = (
        Path(original_filename)
        .suffix
        .lower()
    )


    if not extension:

        extension = ".png"


    saved_filename = (
        f"{uuid.uuid4()}{extension}"
    )


    saved_path = (
        UPLOAD_DIR / saved_filename
    )


    # -----------------------------------------------------
    # Save uploaded image
    # -----------------------------------------------------

    try:

        with saved_path.open("wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Could not save image: {error}"
        )


    # -----------------------------------------------------
    # Open image
    # -----------------------------------------------------

    try:

        image = Image.open(
            saved_path
        ).convert("RGB")

    except Exception as error:

        if saved_path.exists():
            saved_path.unlink()

        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {error}"
        )


    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------

    try:

        image_tensor = (
            transform(image)
            .unsqueeze(0)
            .to(device)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Image preprocessing failed: {error}"
        )


    # -----------------------------------------------------
    # Model prediction
    # -----------------------------------------------------

    try:

        with torch.no_grad():

            outputs = model(
                image_tensor
            )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            predicted_index = (
                torch.argmax(
                    probabilities,
                    dim=1
                ).item()
            )


            confidence = (
                probabilities[
                    0
                ][
                    predicted_index
                ].item()
                * 100
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {error}"
        )


    # -----------------------------------------------------
    # Prediction class
    # -----------------------------------------------------

    prediction = CLASS_NAMES[
        predicted_index
    ]


    # -----------------------------------------------------
    # Probabilities
    # -----------------------------------------------------

    normal_probability = (
        probabilities[0][0].item()
        * 100
    )


    ischemia_probability = (
        probabilities[0][1].item()
        * 100
    )


    bleeding_probability = (
        probabilities[0][2].item()
        * 100
    )


    # -----------------------------------------------------
    # LangGraph explanation
    # -----------------------------------------------------

    try:

        graph_result = assistant_graph.invoke({

            "prediction": prediction,

            "confidence": confidence,

            "explanation": ""

        })


        explanation = graph_result.get(
            "explanation",
            "No explanation generated."
        )

    except Exception as error:

        explanation = (
            f"The model classified this CT image "
            f"as {prediction} with "
            f"{confidence:.2f}% confidence."
        )

        print(
            "LangGraph error:",
            error
        )


    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print()
    print("========================================")
    print("NEW CT IMAGE ANALYSIS")
    print("========================================")
    print("File:", original_filename)
    print("Prediction:", prediction)
    print("Confidence:", f"{confidence:.2f}%")
    print(
        "Normal:",
        f"{normal_probability:.2f}%"
    )
    print(
        "Ischemia:",
        f"{ischemia_probability:.2f}%"
    )
    print(
        "Bleeding:",
        f"{bleeding_probability:.2f}%"
    )
    print("========================================")
    print()


    # -----------------------------------------------------
    # Return response
    # -----------------------------------------------------

    return {

        "success": True,

        "prediction": prediction,

        "confidence": round(
            confidence,
            2
        ),

        "probabilities": {

            "Normal": round(
                normal_probability,
                2
            ),

            "Ischemia": round(
                ischemia_probability,
                2
            ),

            "Bleeding": round(
                bleeding_probability,
                2
            )

        },

        "explanation": explanation,

        "filename": saved_filename

    }