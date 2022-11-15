"""
Date Created: 10/04/2022
Last Modified: 10/09/2022
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
from PIL import Image

#=========================Importing the model=================================
model = tf.keras.models.load_model('fixed_last_layer.hdf5')

#=============================Constant Variables==================================
IMAGE_SIZE = (150, 150)

#==================================Functions==========================================
def image_conversion(image):
    """
    Inputs: image (jpeg or png)
    Outputs: data array
    Creation Date: 10/04/2022
    Purpose:
        Processes a provided image so that it can be input into a model.
    """

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Converts image to RGB. Better for model.
    image = cv2.resize(image, IMAGE_SIZE) #Resizes the image to a standard size for the model.

    st.write(image.shape)

    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

    return image

def run_model(image, model):
    """
    Inputs: image, tensoflow model (.hdf5 file)
    Outputs: np array
    Creation Date: 10/04/2022
    Purpose: Takes a provided image and runs it through the provided tensorflow machine learning model.
    """
    conversion = image_conversion(image)
    st.write(conversion.shape)
    return model.predict(conversion)

def edit_prediction(prediction):
    prediction[0] = 'Benign'
    prediction[1] = 'Malignant'

#================================Webpage Code====================================
#Text to display on the webpage.
st.title('BioScan')
st.subheader('An image classifier.\n\n')
st.sidebar.subheader('Disclaimer')
st.sidebar.write('This is a program created by four computer science students who are not capable of giving valid medical advaice.')
st.sidebar.write('Please do not use this site for valid medical advice. ')
st.sidebar.write('If you are in doubt about your health see a doctor!')
st.header('Instructions')
st.write('To use BioScan take a picture of your skin condition and save it as a jpg or png.')
st.write('Import the image below.')

#Prompts user to upload a file.
options = ['File Upload', 'WebCam Upload']
option = st.radio('Select an option:', options)
if option == 'File Upload':
    user_image = st.file_uploader("Please upload an image file", type = ["jpg", "png"])
else:
    user_image = st.camera_input("Take a picture")

if user_image:
    st.write(user_image.name)

#Code for processing the image. 
if user_image is None:
    #No image provided
    st.text("Please upload an image file!")
else:
    #If image is provided
    if st.button("Acknowledge Disclaimer"):
        image = Image.open(user_image) #opens the image-like element uploades as an image
        st.image(image, use_column_width=True) #displays the provided image on the web app
        prediction = run_model(image, model) #runs the image through the model stores as in variable prediction

        #Gives answer based on prediction.
        #I guessed how the output would look. We should discuss together.
        if np.argmax(prediction) == 0:
            st.write("Our image clasifier labeled it as 'Benign'")
            st.subheader("Remember:")
            st.write("This is an image classifier created by four students.")
            st.write("It was created for a class project.")
            st.write("It should not be taken as valid medical advice.")
            st.write("If you are concerned about your condition see a doctor!")
        else:
            st.header("Results:")
            st.write("Our image classifier labeled it as 'Malignant'")
            st.header("Remember:")
            st.write("This is an image classifier created by four computer science students.")
            st.write("It was created for a class project.")
            st.write("It should not be taken as valid medical advice.")
            st.write("If you are concerned about your condition see a doctor!")
        #Shold present the prediction as a nice little table of % likelihoods. 
        st.write(prediction)

    else:
        st.header('Acknowledge Disclaimer')
        st.write('Before your results are processed, please acknowledge that you understand that this program does not provide valid medical advice.')
        st.write("It is a program writen by four computer science students who have no medical experience. The project is more of a proof of concept.")
        st.write('No classifications provided by this program should be treated as medical advice!')
        st.write("If you are concerned about your health, visit a doctor.")
