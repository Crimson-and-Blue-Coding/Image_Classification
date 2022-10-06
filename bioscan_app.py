"""
Author(s):
    Cody Ptacek
Date Created: 10/04/2022
Last Modified: 10/04/2022
Purpose:
    The web app file for BioScan. Uses streamlit. 
    Application takes an image from user and classifies the category of skin condition that 
        it belongs to by running it through our model. Provides additional info and links on
        the skin condition as well.
"""

#===================================Imports=======================================
#Library Imports
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageOps

#=========================Importing the model=================================
model = tf.keras.models.load_model('First_Model.hdf5')

#=============================Constant Variables==================================
IMAGE_SIZE = (150, 150)

#==================================Functions==========================================
def image_conversion(image):
    """
    Author: Cody Ptacek
    Inputs: image (jpeg or png)
    Outputs: data array
    Creation Date: 10/04/2022
    Purpose:
        Processes a provided image so that it can be input into a model.
    """
    size = (150,150)

    #image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Converts image to RGB. Better for model.
    image = cv2.resize(image, IMAGE_SIZE) #Resizes the image to a standard size for the model.
    #image = np.array(image, dtype= 'float32') #Honestly, not sure what this does or if it is needed...

    st.write(image.shape)

    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

    return image

def run_model(image, model):
    """
    Author: Cody Ptacek
    Inputs: image, tensoflow model (.hdf5 file)
    Outputs: np array?
    Creation Date: 10/04/2022
    Purpose: Takes a provided image and runs it through the provided tensorflow machine learning model.
    """
    conversion = image_conversion(image)
    st.write(conversion.shape)
    return model.predict(conversion)

#================================Webpage Code====================================
#Text to display on the webpage.
st.write('# BioScan')
st.write('An image classifier.\n\n')
st.write('To use BioScan take a picture of your skin condition and save it as a jpg or png.')
st.write('Import the image bellow.')

#Prompts user to upload a file.
user_image = st.file_uploader("Please upload an image file", type = ["jpg", "png"])

#Code for processing the image. 
if user_image is None:
    #No image provided
    st.text("Please upload an image file!")
else:
    #If image is provided
    image = Image.open(user_image) #opens the image-like element uploades as an image
    st.image(image, use_column_width=True) #displays the provided image on the web app
    prediction = run_model(image, model) #runs the image through the model stores as in variable prediction

    #Gives answer based on prediction.
    #I guessed how the output would look. We should discuss together.
    if np.argmax(prediction) == 0:
        st.write("It is likely benign")
    else:
        st.write("It is likely malignant! See a doctor!")

    #Shold present the prediction as a nice little table of % likelihoods. 
    st.write(prediction)
