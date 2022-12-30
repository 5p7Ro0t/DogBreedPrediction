from helper import get_pred_label, create_data_batches, load_model

import streamlit as st
import matplotlib.pyplot as plt
import os
from PIL import Image

st.title("Dog Breed Prediction")
if st.button("Click me"):
    st.write("Hii There")


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('static/images', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


uploaded_file = st.file_uploader("Upload Image")
# text over upload button "Upload Image"
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the image
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        custom_path = "static/images/"
        custom_image_path = [custom_path +
                             fname for fname in os.listdir(custom_path)]
        # custom_image_path

        os.remove("static/images/"+uploaded_file.name)
        custom_data = create_data_batches(custom_image_path, test_data=True)

        full_model = load_model("static/model.h5")
        custom_data
        
        full_model
        # Make predictions on the custom data
        custom_preds = full_model.predict(custom_data)

        # Get custom image prediction labels
        custom_pred_labels = [get_pred_label(
            custom_preds[i]) for i in range(len(custom_preds))]

        # Get custom images (our unbatchify() function won't work since there aren't labels... maybe we could fix this later)
        custom_images = []
        # loop through unbatched data
        for image in custom_data.unbatch().as_numpy_iterator():
            custom_images.append(image)

        # Check custom image predictions
        plt.figure(figsize=(10, 10))
        for i, image in enumerate(custom_images):
            plt.subplot(1, 6, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.title(custom_pred_labels[i])
            plt.imshow(image)
