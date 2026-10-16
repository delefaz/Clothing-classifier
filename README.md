
# Clothing Image Classification System

## Project Overview
This project develops a clothing image classification system using deep learning and MobileNetV2 transfer learning.

The model classifies clothing images into predefined categories using a trained neural network.

## Objectives
- Develop a deep-learning-based clothing classifier.
- Apply MobileNetV2 transfer learning.
- Evaluate model performance using classification metrics.
- Visualise training and validation loss, class distribution, and the confusion matrix.

## Technologies
- Python
- Google Colab
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit (if applicable)

## Model Evaluation
Report the measured results for:
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-score

## Project Files
- `clothing_image_classification.ipynb` — model training and evaluation.
- `fashion_clothing_classifier.keras` — trained model, if included.
- `app.py` — Streamlit application, if included.
- `requirements.txt` — Python dependencies, if included.

## How to Run
1. Clone or download this repository.
2. Install the required Python packages.
3. Open the notebook in Google Colab or Jupyter.
4. Follow the notebook instructions to train or evaluate the model.

## Dataset
The dataset was obtained from the Fashion Product Images (Small) collection available through Hugging Face and comprises labelled fashion product images. Five target categories were selected: T-shirts, shirts, jeans, dresses and shorts. Images were resized to 160 × 160 pixels and represented using three colour channels.

## Limitations
Performance may vary on real-world photographs that differ from the training dataset.

## Author
Bamidele Adeniyi
