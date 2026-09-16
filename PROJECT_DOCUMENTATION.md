# AI-Based Brain CT Stroke Classification and Assistance System

## 1. Project Overview

This project is an AI-based system for classifying brain CT images into three categories:

- Normal
- Ischemia
- Bleeding

The system uses a ResNet18 deep learning model for image classification and LangGraph for conditional assistant responses.

## 2. Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- NumPy
- Pandas
- Scikit-learn
- Pillow
- LangGraph
- Matplotlib

## 3. System Workflow

1. Input brain CT image.
2. Resize and normalize the image.
3. Pass the image to the trained ResNet18 model.
4. Generate the predicted class and confidence score.
5. Send the prediction to LangGraph.
6. Route the prediction to the relevant explanation node.
7. Generate a class-specific assistant response.

## 4. Classification Classes

| Label | Class |
|---|---|
| 0 | Normal |
| 1 | Ischemia |
| 2 | Bleeding |

## 5. Deep Learning Model

The project uses ResNet18 with the final fully connected layer modified for three output classes.

The trained model is stored in:

models/brain_ct_resnet18_weighted.pth

The model was trained using a more balanced training dataset.

## 6. Model Performance

The balanced model achieved approximately:

- Test Accuracy: 90.68%

The balanced training approach improved the classification of the Ischemia class.

## 7. LangGraph Workflow

LangGraph uses conditional routing based on the model prediction.

Possible routes:

- Normal → Normal explanation node
- Ischemia → Ischemia explanation node
- Bleeding → Bleeding explanation node

## 8. Project Structure

```text
Brain_CT_Stroke_AI/
├── dataset/
├── src/
├── models/
├── notebooks/
├── app/
├── check_dataset.py
├── requirements.txt
└── PROJECT_DOCUMENTATION.md