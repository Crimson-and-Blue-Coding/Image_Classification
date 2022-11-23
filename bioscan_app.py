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
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import pandas as pd

#Class Import
from conditions import Conditions

#=========================Importing the model=================================
model = tf.keras.models.load_model('fixed_last_layer.hdf5')

#=============================Constant Variables==================================
IMAGE_SIZE = (150, 150)
basic = '<p style="font-family:Georgia; color:Black; font-size: 24px;">'
basicbold = '<p style="font-family:Georgia; font-weight: bold; color:Black; font-size: 24px;">'
header = '<p style="font-family:Georgia; color:Black; font-size: 44px;">'
subheader = '<p style="font-family:Georgia; font-weight: bold; color:Black; font-size: 24px;">'
title = '<p style="font-family:Georgia; font-weight: bold; color:Black; font-size: 58px;">'
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

    #st.write(image.shape)

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
    #st.write(conversion.shape)
    return model.predict(conversion)

def edit_prediction(prediction):
    prediction[0] = 'Benign'
    prediction[1] = 'Malignant'

#================================Webpage Code====================================
#Text to display on the webpage.
import base64
def background(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
background('background_img.jpg')   
st.markdown(title + 'BioScan' + "</p>", unsafe_allow_html=True)
st.markdown(header + 'An image classifier.\n\n' + "</p>", unsafe_allow_html=True)
st.sidebar.subheader('Disclaimer')
st.sidebar.write('This is a program created by four computer science students who are not capable of giving valid medical advaice.')
st.sidebar.write('Please do not use this site for valid medical advice. ')
st.sidebar.write('If you are in doubt about your health see a doctor!')
st.markdown(header + 'Instructions:' + "</p>", unsafe_allow_html=True)
st.markdown(subheader + 'To use BioScan take a picture of your skin condition and save it as a jpg or png.' + "</p>", unsafe_allow_html=True)
st.markdown(basic + 'Import the image below.' + "</p>", unsafe_allow_html=True)

#Prompts user to upload a file.
options = ['File Upload', 'WebCam Upload']
option = st.radio('Select an option:', options)
if option == 'File Upload':
    user_image = st.file_uploader("Please upload an image file", type = ["jpg", "png"])
else:
    user_image = st.camera_input("Take a picture")

if user_image:
    st.write(user_image.name)

#optional survery so user is able to add symptoms they are experiencing to create more accurate results

counter = 0
st.markdown(basic + 'Complete the optional Survey below based on the symptoms you are experiencing:' + "</p>", unsafe_allow_html=True)
options = st.multiselect( " ",
    ['Itchy', 'Painful', 'Occasionally bleeds', 'Burning sensation', 'Quickly developed', 'Oozing', 'Scaliness', 'Raised','Change in Sensation', 'None of the Above'])

for i in range(len(options)):
    if options[i] == "Itchy":
        counter = counter+1
    if options[i] == "Painful":
        counter = counter+2
    if options[i] == "Occasionally bleeds":
        counter = counter+3
    if options[i] == "Burning sensation":
        counter = counter+2
    if options[i] == 'Quickly developed':
        counter = counter+2
    if options[i] == 'Raised':
        counter = counter+2
    if options[i] == 'Change in Sensation':
        counter = counter+2
    if options[i] == 'Oozing':
        counter = counter+2
    if options[i] == 'Scaliness':
        counter = counter+2
    if options[i] == 'None of the Above':
        counter = 0

percent = int((counter/18)*100)
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
        condition = Conditions()
        condition.generateResults(np.argmax(prediction))
        if condition.conditionName == "Benign":
            st.balloons()
        
        basic = '<p style="font-family:Georgia; color:Black; font-size: 24px;">'
        st.markdown(basic + f"Our image clasifier labeled it as '{condition.conditionName}'" +"</p>", unsafe_allow_html=True)
        st.markdown(subheader + 'More Information:' + "</p>", unsafe_allow_html=True)
        #st.write(f"{condition.conditionName}")
        st.markdown(basicbold + f'{condition.conditionName}' + "</p>", unsafe_allow_html=True)
        st.markdown(basic + f'{condition.description}' +"</p>", unsafe_allow_html=True)
        #st.write(condition.description)
        st.markdown(basicbold + 'Related Links:' + "</p>", unsafe_allow_html=True)
        for name,link in condition.links:
            st.write(f"[{name}]({link})")
        st.markdown(subheader + 'Remember:' + "</p>", unsafe_allow_html=True)
        st.markdown(basic + "This is an image classifier created by four students." +"</p>", unsafe_allow_html=True)
        st.markdown(basic + "It was created for a class project." +"</p>", unsafe_allow_html=True)
        st.markdown(basic + "It should not be taken as valid medical advice" +"</p>", unsafe_allow_html=True)
        st.markdown(basic + "If you are concerned about your condition see a doctor!" +"</p>", unsafe_allow_html=True) 

        outOf = int((len(options)/9)*100)

        if percent >= 30 or options.count('Occasionally bleeds') == 1:
            st.markdown(basic + f"You have {outOf}% of the symptoms in the survey:" +"</p>", unsafe_allow_html=True) 
            st.markdown(basic + "Because of this, if you are still worried about your condition even after the results, it is recommended to see a doctor!" +"</p>", unsafe_allow_html=True) 
            
        #Presents the prediction as a nice little table of % likelihoods. 
        prediction_dict = {'Benign': [prediction[0][0]], 'Malignant': [prediction[0][1]]} #Makes a dictionary with keys of conditions and results from the model's output
        st.table(pd.DataFrame.from_dict(prediction_dict)) #Prints a table of the results on the website.

        benign_percent = prediction[0][0]
        malignant_percent = prediction[0][1]
        labels = ['Benign', 'Malignant']
        sizes = [benign_percent, malignant_percent]

        fig1, ax1 = plt.subplots()
        explode = (0, 0.1)
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

        ax1.axis('equal')

        st.pyplot(fig1)

        st.markdown(subheader + 'Should you seek medical help?' + "</p>", unsafe_allow_html=True)
        if condition.conditionName == 'Malignant':
            st.markdown(basic + "Because of the image results being Malignant, it is highly recommended to see a doctor!" + "</p>", unsafe_allow_html=True)
        elif percent >= 30 and condition.conditionName == 'Benign':
            st.markdown(basic + "Because of the severity of the symptoms you chose in the survery, if you are still worried about your condition even after the Benign results, it is highly recommended to see a doctor!" + "</p>", unsafe_allow_html=True)
        elif condition.conditionName == 'Benign' and options.count('None of the Above') == 1:
            st.markdown(basic + "You have no symptoms and your image results are Benign, but see a doctor if you are still concerned." + "</p>", unsafe_allow_html=True)
        else:
            st.markdown(basic + "If you are still concerned after results, see a doctor." + "</p>", unsafe_allow_html=True)
    else:
        st.markdown(subheader + 'Acknowledge Disclaimer' + "</p>", unsafe_allow_html=True)
        st.markdown(basic + 'Before your results are processed, please acknowledge that you understand that this program does not provide valid medical advice.' + "</p>", unsafe_allow_html=True)
        st.markdown(basic + 'It is a program writen by four computer science students who have no medical experience. The project is more of a proof of concept.' + "</p>", unsafe_allow_html=True)
        st.markdown(basic + 'No classifications provided by this program should be treated as medical advice!' + "</p>", unsafe_allow_html=True)
        st.markdown(basic + 'If you are concerned about your health, visit a doctor.' + "</p>", unsafe_allow_html=True)
