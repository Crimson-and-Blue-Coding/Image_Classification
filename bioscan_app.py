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
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
from io import BytesIO

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
    return model.predict(conversion)

def edit_prediction(prediction):
    prediction[0] = 'Benign'
    prediction[1] = 'Malignant'

def graph_percentages(prediction):
    labels = ['Malignant', 'Benign']
    malig_percent = prediction[0][1]*100
    benign_percent = prediction[0][0]*100
    sizes = [malig_percent, benign_percent]
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.pie(sizes, labels=labels, shadow=True, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    
    buf = BytesIO()
    fig1.savefig(buf, format="png")
    return buf

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
    user_image_list = st.file_uploader("Please upload an image file", type = ["jpg", "png"], accept_multiple_files=True)
else:
    user_image = st.camera_input("Take a picture")
    user_image_list = []
    if user_image:
        user_image_list.append(user_image)


#Code for processing the image. 
if len(user_image_list) == 0:
    #No image provided
    st.text("Please upload an image file!")
else:
    #If image is provided
    if st.button("Acknowledge Disclaimer"):
        image_predictions = []
        for user_image in user_image_list:
            img_data = dict()

            image = Image.open(user_image) #opens the image-like element uploades as an image
            prediction = run_model(image, model) #runs the image through the model stores as in variable prediction
            img_data['image'] = image
            img_data['prediction'] = prediction

            if np.argmax(prediction) == 0:
                img_data['result'] = 'Benign'
            else:
                img_data['result'] = 'Malignant'

            image_predictions.append(img_data)

        for image_results in image_predictions:
            image = image_results['image']
            image.thumbnail((400, 400), Image.ANTIALIAS)
            # image = image.resize((400, 400))
            graph = graph_percentages(image_results['prediction'])
            graph = Image.open(graph)
            graph.thumbnail((300, 300))
            st.image([image, graph])
            st.write(f"Result: {image_results['result']}")


        #st.image(image, use_column_width=True) #displays the provided image on the web app
        #Gives answer based on prediction.
        #I guessed how the output would look. We should discuss together.
                # st.write("Our image clasifier labeled it as 'Benign'")

                # st.header("Results:")
                # st.write("Our image classifier labeled it as 'Malignant'")

        st.header("Remember:")
        st.write("This is an image classifier created by four computer science students.")
        st.write("It was created for a class project.")
        st.write("It should not be taken as valid medical advice.")
        st.write("If you are concerned about your condition see a doctor!")
    else:
        st.header('Acknowledge Disclaimer')
        st.write('Before your results are processed, please acknowledge that you understand that this program does not provide valid medical advice.')
        st.write("It is a program writen by four computer science students who have no medical experience. The project is more of a proof of concept.")
        st.write('No classifications provided by this program should be treated as medical advice!')
        st.write("If you are concerned about your health, visit a doctor.")