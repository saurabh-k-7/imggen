import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from io import BytesIO  # Required for converting images for download

# Importing functions from models
from models.image_generation import generate_image  # Image generation logic
from models.image_classification import classify_image  # Image classification logic

# Adding background image or video (placeholder for now)
page_bg_image = """
<style>
[data-testid="stSidebar"] {
    background-color: rgba(30, 30, 30, 0.9);
    color: white;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
h1, h2, h3, h4, h5, h6 {
    color: #f5f5f5;
    font-family: 'Arial Black', sans-serif;
}
</style>
"""
st.markdown(page_bg_image, unsafe_allow_html=True)

# Sidebar for user interaction
st.sidebar.title("🎨 App Navigation")
option = st.sidebar.selectbox(
    "Choose an Action:",
    ("✨ Image Generation", "🖼️ Image Classification"),
)

# Styling for headers
st.markdown(
    """
    <style>
    .title-style {
        font-size: 50px;
        font-weight: bold;
        color: #FFFFFF;
        text-shadow: 2px 2px 5px #000000;
    }
    .subtitle-style {
        font-size: 20px;
        color: #D3D3D3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Ensure the `temp` directory exists
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Image Generation Section
if option == "✨ Image Generation":
    st.markdown('<h1 class="title-style">Generate Stunning Images</h1>', unsafe_allow_html=True)

    # Input for prompt
    prompt = st.text_input(
        "💡 Enter your creative prompt:",
        "A scenic view of the mountains at sunrise",
    )

    # Button to generate image
    if st.button("🎨 Generate Image"):
        with st.spinner("✨ Generating your masterpiece... please wait!"):
            generated_image = generate_image(prompt)  # Call the image generation function

        # Check if the image was generated successfully
        if generated_image:
            st.success("🎉 Image generated successfully!")
            st.image(generated_image, caption="Generated Image", use_container_width=True)

            # Convert PIL image to bytes for download
            buffered = BytesIO()
            generated_image.save(buffered, format="PNG")
            buffered.seek(0)

            # Add a download button
            st.download_button(
                label="💾 Download Image",
                data=buffered,
                file_name="generated_image.png",
                mime="image/png",
            )
        else:
            st.error("⚠️ Failed to generate the image. Please try again.")

# Image Classification Section
elif option == "🖼️ Image Classification":
    st.markdown('<h1 class="title-style">Classify Your Images</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subtitle-style">Discover the emotions in your photos!</h3>', unsafe_allow_html=True)

    # Upload image file
    uploaded_file = st.file_uploader(
        "📤 Upload an image to classify:",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded_file:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Save the uploaded file temporarily
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Classify the image
        if st.button("🔍 Classify Image"):
            with st.spinner("🤔 Analyzing the image... please wait!"):
                result = classify_image(temp_path)  # Call the classification function
                st.success(f"Prediction: {result}")

        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
