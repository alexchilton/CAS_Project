# Fruit & Vegetable Image Classification — CAS AML Project 1
**University of Bern — CAS in Advanced Machine Learning, Modules 1 & 2**

A computer vision pipeline for classifying 36 produce categories using CNNs and transfer learning, with a FastAPI prediction service and webcam integration. A key finding of the project was discovering and resolving severe data contamination in the source dataset before any meaningful results could be obtained.

---

## Overview

The goal was to build a model that identifies fruits and vegetables from images and (as a stretch goal) connects to the OpenAI API to suggest recipes using the identified ingredients.

**36 classes:** 10 fruits (banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango) + 26 vegetables.

---

## The Data Problem

Before any modelling, the dataset required significant cleaning:

- **Test and validation sets were identical** and also overlapped with training data
- Duplicate detection via image hashing (PIL + imagehash) identified and removed contaminated samples
- Several near-duplicate categories were merged (sweetcorn/corn, bell pepper/capsicum)
- **Impact:** Initial inflated results (97% Random Forest accuracy) collapsed to a meaningful 43% baseline post-cleanup — revealing exactly how misleading benchmark numbers can be without proper data hygiene

**Final cleaned dataset:** ~2,362 training / ~302 validation images

---

## Models

| Model | Notes |
|-------|-------|
| Random Forest | Non-deep-learning baseline for comparison |
| Custom CNN | With data augmentation (flip, rotation, zoom) |
| InceptionV3 | Transfer learning, 299×299 input |
| ResNet-50 | Transfer learning, 224×224 input — used in production service |

All transfer learning models used learning rate scheduling and model checkpointing.

---

## Architecture

```
Webcam / Image → FastAPI Service → ResNet-50 → Classification
                                                     ↓
                                           (Recipe suggestion — planned)
```

**FastAPI Service** (`fastapi_image_predictor_service.py`):
- POST `/predict/` endpoint
- Accepts image upload, returns JSON classification

**Webcam Capture** (`capture_image.py`):
- OpenCV video capture
- Manual frame selection, 128×128 resize
- Sends to local prediction endpoint

---

## Key Findings

- Transfer learning (ResNet-50, InceptionV3) substantially outperformed the custom CNN after data cleaning
- Webcam-based prediction worked but was impractical — hands obstructing produce during capture introduced noise ("the model has been known to identify me as a potato")
- A mobile app interface would better suit real-world use

---

## Repository Structure

```
├── final_notebook.ipynb              # Main training and evaluation notebook
├── pre_traintest_fileoperation.ipynb # Dataset deduplication and cleanup
├── fastapi_image_predictor_service.py # Prediction API
├── capture_image.py                  # Webcam integration
├── functions_m1_m2.py               # Shared utilities (loading, preprocessing, visualisation)
└── requirements.txt
```

---

## Getting Started

```bash
pip install -r requirements.txt

# Run the prediction service
uvicorn fastapi_image_predictor_service:app --reload

# In another terminal — webcam capture
python capture_image.py
```

---

## Course Context

**Programme:** CAS in Advanced Machine Learning, University of Bern — Modules 1 & 2  
**Key Libraries:** TensorFlow/Keras, scikit-learn, FastAPI, OpenCV, Pillow, MLflow
