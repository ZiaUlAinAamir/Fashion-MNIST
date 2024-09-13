import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Title
st.title('Fashion MNIST CNN Model')

# Load the model
def load_cnn_model():
    model = load_model("model.keras")
    return model

# Load the CNN model
model = load_cnn_model()
st.sidebar.success("Model Loaded Successfully!")

# Upload an image to make a prediction
uploaded_file = st.file_uploader("Upload an Image of Fashion Item", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    img = Image.open(uploaded_file).convert("L")  # Convert image to grayscale
    st.image(img, caption="Uploaded Image", use_column_width='always')

    # Preprocess the image to match the input shape of the model
    img = img.resize((28, 28))  # Fashion MNIST images are 28x28 pixels
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (for grayscale)

    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Labels for Fashion MNIST dataset
    class_labels = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    st.write(f"Predicted Class: {class_labels[predicted_class]}")
