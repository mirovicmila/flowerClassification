import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('/content/my_model1.hdf5')
  return model

with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Flower Classification
         """
         )

file = st.file_uploader("Please upload an image", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        #size = (180,180)    
        #image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        #image = np.asarray(image)
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #img_reshape = img[np.newaxis,...]

        img_array = tf.keras.utils.img_to_array(image_data)
        img_array = tf.expand_dims(img_array, 0)
    
        prediction = model.predict(img_array)
        
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    image = image.resize((180,180))
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    st.write(score)
    class_names=['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    st.write("This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
