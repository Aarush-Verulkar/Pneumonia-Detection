import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


# Include custom CSS using st.markdown
st.markdown(
    """
    <style>
    @import url('style.css');

    /* Define a custom class for headers and titles */
    .custom-text {
        font-family: 'JetBrains Mono'; 
        font-weight: bold;
        font-size: 24px; 
        color:#F99417;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Your Streamlit app code goes here

set_background('./bgs/bg5.png')

# set title with the custom class
st.markdown('<p class="custom-text">Pneumonia Detection</p>', unsafe_allow_html=True)

# set header with the custom class
st.markdown('<p class="custom-text">Upload Your X-Ray:</p>', unsafe_allow_html=True)

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/pneumonia_classifier.h5')

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
