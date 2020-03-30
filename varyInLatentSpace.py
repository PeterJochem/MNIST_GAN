import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import (Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import load_model
import os
import cv2
from PIL import Image
import random

# Load the generator and the discriminator
discriminator = load_model('./models/discriminator.h5')
generator = load_model('./models/generator.h5')

# Turn all the images we have created as we varied the input 
# in latent space into a video
def generateVideo():
    image_folder = '.' # make sure to use your folder 
    video_name = 'latentSpaceVariation.avi'
    os.chdir("latentSpace")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # setting the frame width, height width 
    # the width, height of first image 
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    # Add all the images to the video 
    for i in range(len(images)):

        imageName = str(i) + ".jpeg"

        video.write(cv2.imread(os.path.join(image_folder, imageName)))

    cv2.destroyAllWindows()
    video.release()


imageIndex = 0
numIterations = 100

latentVector = np.random.normal(0, 1, (1, 100) )

for i in range(numIterations):
    
    next_evolution = generator.predict(latentVector)

    randomIndex = int(random.random() * len(latentVector[0]) )
    

    # Update the latent vector
    latentVector[0][randomIndex] = latentVector[0][randomIndex] * 0.5 

    # Rescale the pixel values into [0, 1]
    next_evolution = 0.5 * next_evolution + 0.5

    # Reshape the image into a 28 x 28 array
    next_evolution = np.reshape(next_evolution, (28, 28))

    # Convert the image entries in [0, 1] into [0, 255]
    next_evolution = next_evolution * 255

    # Convert from numpy array to an image format
    img = Image.fromarray(next_evolution)

    img = img.convert("L")
    img.save( "latentSpace/" + str(imageIndex) + '.jpeg')
   
    imageIndex = imageIndex + 1

generateVideo()


