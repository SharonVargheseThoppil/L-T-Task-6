import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CIFAR-10 CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cifar10_cnn_model.keras"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}"
        )

    return tf.keras.models.load_model(model_path)


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):
    """
    Convert image to RGB,
    resize to 32x32,
    normalize pixel values,
    and add batch dimension.
    """

    image = image.convert("RGB")

    image = image.resize((32, 32))

    image_array = np.array(image)

    image_array = image_array.astype("float32") / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# =========================================================
# HEADER
# =========================================================

st.title("🧠 CIFAR-10 Image Classifier")

st.markdown(
    """
    ### Deep Learning Image Prediction using Streamlit

    Upload an image and use the trained **CIFAR-10 CNN model**
    to predict its class.
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Dataset:** CIFAR-10")
    st.write("**Model:** Convolutional Neural Network")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Interface:** Streamlit")
    st.write("**Input Size:** 32 × 32 pixels")

    st.divider()

    st.subheader("📚 Supported Classes")

    for class_name in CLASS_NAMES:
        st.write(f"• {class_name.capitalize()}")


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_model()

    st.success("✅ Model loaded successfully.")

except Exception as e:

    st.error("❌ Unable to load the trained model.")

    st.exception(e)

    st.stop()


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.subheader("📤 Upload an Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# IF IMAGE IS UPLOADED
# =========================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        # -------------------------------------------------
        # DISPLAY IMAGE AND INFORMATION
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("🖼️ Uploaded Image")

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            st.subheader("📋 Image Information")

            st.write(
                f"**File name:** {uploaded_file.name}"
            )

            st.write(
                f"**Original size:** "
                f"{image.size[0]} × {image.size[1]} pixels"
            )

            st.write(
                f"**Image mode:** {image.mode}"
            )

            st.info(
                "The image will be resized to 32 × 32 pixels "
                "before prediction."
            )

        st.divider()

        # -------------------------------------------------
        # PREDICTION BUTTON
        # -------------------------------------------------

        predict_button = st.button(
            "🔮 Predict Image",
            type="primary",
            use_container_width=True
        )

        if predict_button:

            with st.spinner("Analyzing image..."):

                # Preprocess image
                processed_image = preprocess_image(image)

                # Make prediction
                predictions = model.predict(
                    processed_image,
                    verbose=0
                )

                # Find predicted class
                predicted_index = int(
                    np.argmax(predictions[0])
                )

                predicted_class = CLASS_NAMES[
                    predicted_index
                ]

                # Get confidence
                confidence = float(
                    predictions[0][predicted_index]
                )

            # -------------------------------------------------
            # PREDICTION RESULT
            # -------------------------------------------------

            st.divider()

            st.subheader("🎯 Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    label="Predicted Class",
                    value=predicted_class.upper()
                )

            with result_col2:

                st.metric(
                    label="Confidence",
                    value=f"{confidence * 100:.2f}%"
                )

            # -------------------------------------------------
            # CONFIDENCE
            # -------------------------------------------------

            st.write("### 📈 Confidence Level")

            st.progress(confidence)

            # -------------------------------------------------
            # TOP 5 PREDICTIONS
            # -------------------------------------------------

            st.write("### 📊 Top 5 Prediction Probabilities")

            top_indices = np.argsort(
                predictions[0]
            )[::-1][:5]

            for index in top_indices:

                class_name = CLASS_NAMES[int(index)]

                probability = float(
                    predictions[0][index]
                )

                st.write(
                    f"**{class_name.capitalize()}** — "
                    f"{probability * 100:.2f}%"
                )

                st.progress(probability)

            # -------------------------------------------------
            # INTERPRETATION
            # -------------------------------------------------

            st.divider()

            st.subheader("📝 Result Interpretation")

            st.success(
                f"The model predicts that the uploaded image "
                f"belongs to the **{predicted_class.upper()}** "
                f"class with a confidence of "
                f"**{confidence * 100:.2f}%**."
            )

    except Exception as e:

        st.error(
            "❌ There was a problem processing the uploaded image."
        )

        st.exception(e)


# =========================================================
# NO IMAGE UPLOADED
# =========================================================

else:

    st.info(
        "👆 Please upload a JPG, JPEG, or PNG image "
        "to begin prediction."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Task 6: Creating a Streamlit User Interface | "
    "L&T EduTech Deep Learning Project"
)