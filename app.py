
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path




# 1. APP CONFIGURATION

st.set_page_config(
    page_title="Clothing Classifier",
    page_icon="👕",
    layout="centered"
)

st.title("👕 Clothing Classifier")
st.write(
    "Upload a clothing image and this trained model "
    "will predict its category."
)




# 2. FIND AND LOAD THE MODEL

# Locate the folder containing this app.py file
BASE_DIR = Path(__file__).resolve().parent

# Model file should be in the same folder as app.py
MODEL_PATH = BASE_DIR / "clothing_classifier.keras"


@st.cache_resource
def load_clothing_model(model_path):
    """Load and cache the trained model."""
    return tf.keras.models.load_model(model_path)


# Stop if the model file cannot be found
if not MODEL_PATH.is_file():
    st.error("The trained model file was not found.")
    st.write("The app is looking for the model here:")
    st.code(str(MODEL_PATH))

    st.info(
        "Download clothing_classifier.keras "
        "from Google Colab and place it in the same "
        "folder as app.py."
    )
    st.stop()

# Load the model safely
try:
    model = load_clothing_model(str(MODEL_PATH))
except Exception as error:
    st.error("The model file exists but could not be loaded.")
    st.write("Check that it is a valid, complete Keras model.")
    st.code(str(error))
    st.stop()




# 3. CLOTHING CATEGORIES

# These names MUST match the exact class order used
# when training the model.
CLASS_NAMES = [
    "Tshirts",
    "Shirts",
    "Jeans",
    "Dresses",
    "Shorts"
]

# Check the model's output size against our categories
output_size = model.output_shape[-1]

if output_size != len(CLASS_NAMES):
    st.error(
        f"The model predicts {output_size} classes, "
        f"but the app defines {len(CLASS_NAMES)} categories. "
        "Update CLASS_NAMES to match the training labels."
    )
    st.stop()




# 4. UPLOAD AN IMAGE

uploaded_file = st.file_uploader(
    "Choose a clothing image",
    type=["jpg", "jpeg", "png"]
)




# 5. PREPROCESS AND PREDICT

if uploaded_file is not None:

    try:
        # Open the uploaded image and convert it to RGB
        image = Image.open(uploaded_file).convert("RGB")

        # Display the original image
        st.image(
            image,
            caption="Uploaded image",
            width=350
        )

        with st.spinner("Analysing the clothing image..."):

            # Resize to the image size used during training
            resized_image = image.resize((160, 160))

            # Convert image to a NumPy array
            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            # Add a batch dimension: (1, 160, 160, 3)
            image_array = np.expand_dims(image_array, axis=0)

            # Predict the clothing category
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

        # Get the highest-scoring category
        best_index = int(np.argmax(predictions))
        best_category = CLASS_NAMES[best_index]
        confidence = float(predictions[best_index])

        


        # 6. DISPLAY RESULTS
       
        st.subheader("Prediction")

        st.success(f"Predicted category: {best_category}")

        st.metric(
            "Model score",
            f"{confidence * 100:.2f}%"
        )

        st.caption(
            "The model score is not necessarily the probability "
            "that the prediction is correct."
        )

        # Show scores for all categories
        st.subheader("Scores for all categories")

        scores = {
            CLASS_NAMES[i]: float(predictions[i])
            for i in range(len(CLASS_NAMES))
        }

        st.bar_chart(scores)

    except Exception as error:
        st.error("The uploaded image could not be processed.")
        st.code(str(error))




# 7. FOOTER

st.divider()

st.caption(
    " Clothing Classifier model by Bamidele Adeniyi (Group 20)"
)
