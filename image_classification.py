import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

print(class_names_label)

IMAGE_SIZE = (150, 150)


#loading the data
def load_data():
    DIRECTORY = r"C:\Users\clee1\miniconda3\envs\tf\BioScan\Image_Classification"
    CATEGORY = ["seg_train", "seg_test"]

    output = []

    for category in CATEGORY:
        path = os.path.join(DIRECTORY, category)
        print(path)
        images = []
        labels = []

        print("Loading {}".format(category))

        for folder in os.listdir(path):
            label = class_names_label[folder]

            for file in os.listdir(os.path.join(path, folder)):

                img_path = os.path.join(os.path.join(path, folder), file)

                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                images.append(image)
                labels.append(label)

        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')

        output.append((images, labels))

    return output

(train_images, train_labels), (test_images, test_labels) = load_data()

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

def display_examples(class_names, images, labels):
    #display 25 images from the images array with its corresponding labels
    figsize = (20, 20)
    fig = plt.figure(figsize=figsize)
    fig.suptitle("Sample images of the dataset", fontsize = 16)
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # image = cv2.resize(images[i], figsize)
        plt.imshow(images[i].astype(np.uint8)) #used to be a float
        plt.xlabel(class_names[labels[i]])
    plt.show()

display_examples(class_names, train_images, train_labels) 

model = tf.keras.Sequential([ 
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=128, epochs=4, validation_split=0.2)

def plot_accuracy_loss(history):
    #plot the accuracy and the loss during training of the nn
    fig=plt.figure(figsize=(10,5))

    #plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'], 'bo--', label='acc')
    plt.plot(history.history['val_accuracy'], 'ro--', label="val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    #plot loss function
    #plt.subplot(222)
    plt.show()



