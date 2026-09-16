from pathlib import Path
from typing import TypedDict
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from langgraph.graph import StateGraph, START, END


# =========================================================
# 1. Project paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "brain_ct_resnet18_weighted.pth"
)


# =========================================================
# 2. Default image path
# =========================================================

DEFAULT_IMAGE_PATH = (
    BASE_DIR
    / "dataset"
    / "data"
    / "Ischemia"
    / "10088.png"
)


# =========================================================
# 3. Command-line image path support
# =========================================================

if len(sys.argv) > 1:

    IMAGE_PATH = Path(sys.argv[1])

    if not IMAGE_PATH.is_absolute():
        IMAGE_PATH = BASE_DIR / IMAGE_PATH

else:
    IMAGE_PATH = DEFAULT_IMAGE_PATH


# =========================================================
# 4. Class names
# =========================================================

CLASS_NAMES = {
    0: "Normal",
    1: "Ischemia",
    2: "Bleeding",
}


# =========================================================
# 5. LangGraph state
# =========================================================

class AssistantState(TypedDict):
    prediction: str
    confidence: float
    explanation: str


# =========================================================
# 6. LangGraph explanation node
# =========================================================

def explain_prediction(state: AssistantState):

    prediction = state["prediction"]
    confidence = state["confidence"]

    if prediction == "Normal":

        explanation = (
            f"The model classified this CT image as Normal "
            f"with {confidence:.2f}% confidence. "
            "No abnormality was detected by the model. "
            "This result should still be reviewed by a qualified "
            "medical professional."
        )

    elif prediction == "Ischemia":

        explanation = (
            f"The model classified this CT image as Ischemia "
            f"with {confidence:.2f}% confidence. "
            "Ischemia may indicate reduced blood flow to brain tissue. "
            "Urgent medical assessment is recommended."
        )

    elif prediction == "Bleeding":

        explanation = (
            f"The model classified this CT image as Bleeding "
            f"with {confidence:.2f}% confidence. "
            "Possible brain bleeding is a medical emergency. "
            "Immediate professional medical assessment is required."
        )

    else:

        explanation = "Unknown prediction."

    return {
        "explanation": explanation
    }


# =========================================================
# 7. Build LangGraph
# =========================================================

graph_builder = StateGraph(AssistantState)

graph_builder.add_node(
    "explain_prediction",
    explain_prediction
)

graph_builder.add_edge(
    START,
    "explain_prediction"
)

graph_builder.add_edge(
    "explain_prediction",
    END
)

assistant_graph = graph_builder.compile()


# =========================================================
# 8. Check model and image files
# =========================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )


if not IMAGE_PATH.exists():

    raise FileNotFoundError(
        f"Image file not found:\n{IMAGE_PATH}"
    )


# =========================================================
# 9. Load trained ResNet18 model
# =========================================================

device = torch.device("cpu")

print("\nLoading model...")
print(f"Model path: {MODEL_PATH}")

model = models.resnet18(weights=None)

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

print("Model loaded successfully.")


# =========================================================
# 10. Image preprocessing
#
# IMPORTANT:
# This must match the training/test pipeline.
#
# Current training/test pipeline uses:
# Resize -> ToTensor
#
# Therefore Normalize is NOT used here.
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =========================================================
# 11. Load image
# =========================================================

print("\n========================================")
print("           IMAGE INFORMATION")
print("========================================")

print(f"Image path: {IMAGE_PATH.resolve()}")
print(f"Image filename: {IMAGE_PATH.name}")
print(f"Folder name: {IMAGE_PATH.parent.name}")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

print(f"Original image size: {image.size}")
print(f"Image mode: {image.mode}")


# =========================================================
# 12. Pixel statistics
# =========================================================

pixel_tensor = transforms.ToTensor()(image)

print(
    f"Pixel minimum: {pixel_tensor.min().item():.4f}"
)

print(
    f"Pixel maximum: {pixel_tensor.max().item():.4f}"
)

print(
    f"Pixel mean: {pixel_tensor.mean().item():.4f}"
)


# =========================================================
# 13. Preprocess image
# =========================================================

input_tensor = (
    transform(image)
    .unsqueeze(0)
    .to(device)
)

print(
    f"Model input shape: {input_tensor.shape}"
)


# =========================================================
# 14. Model prediction
# =========================================================

with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = (
        probabilities[0][predicted_class].item()
        * 100
    )


# =========================================================
# 15. Prediction and probabilities
# =========================================================

prediction = CLASS_NAMES[predicted_class]

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


# =========================================================
# 16. Model debug information
# =========================================================

print("\n========================================")
print("          MODEL DEBUG INFORMATION")
print("========================================")

print(
    f"Raw model output: {outputs}"
)

print(
    f"All probabilities: {probabilities}"
)


# =========================================================
# 17. Send prediction to LangGraph
# =========================================================

result = assistant_graph.invoke({

    "prediction": prediction,

    "confidence": confidence,

    "explanation": ""
})


# =========================================================
# 18. Display final result
# =========================================================

print("\n========================================")
print("       BRAIN CT STROKE PREDICTION")
print("========================================")

print(
    f"Image: {IMAGE_PATH.name}"
)

print(
    f"Actual folder: {IMAGE_PATH.parent.name}"
)

print(
    f"\nPrediction: {prediction}"
)

print(
    f"Confidence: {confidence:.2f}%"
)

print("\nClass Probabilities:")

print(
    f"Normal: {normal_probability:.2f}%"
)

print(
    f"Ischemia: {ischemia_probability:.2f}%"
)

print(
    f"Bleeding: {bleeding_probability:.2f}%"
)


# =========================================================
# 19. LangGraph assistant
# =========================================================

print("\n========================================")
print("          LANGGRAPH ASSISTANT")
print("========================================")

print(
    result["explanation"]
)


# =========================================================
# 20. Safety note
# =========================================================

print(
    "\nNote: This is a research prototype,"
)

print(
    "not a clinical diagnostic system."
)