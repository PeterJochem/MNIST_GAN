import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import (
Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import load_model
import os 
import cv2  
from PIL import Image  

# These define the shape of the MNIST images
num_rows = 28
num_columns = 28

# MNIST images are grayscale
num_channels = 1

# This defines the shape of the image that goes
# into the discriminator network
img_shape = (num_rows, num_columns, num_channels)

# This is the size of the noise vector
noise_vector_length = 100

# Evolution vector
# Store a random noise vector and use it as 
# input to the generator after each training iteration of 
# the generator
evolution_vector = np.random.normal(0, 1, (1, 100) )

imageIndex = 0

# This takes all the images in the evolution directory and 
# turns them into a video called evolution.avi 
def generateVideo(): 
    image_folder = '.' # make sure to use your folder 
    video_name = 'evolution.avi'
    os.chdir("evolution") 
      
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


# This takes the gan and creates image with it and plots them
# into a grid of 4x4 images
# generator is the generator neural network
# grid_rows is the number of rows in the display grid
# and grid columns is the number of columns in the grid columns
# Retuns void
def generate_images(generator, grid_rows = 4, grid_columns = 4):
    
    # Generate random input vector
    z = np.random.normal(0, 1, (grid_rows * grid_columns, noise_vector_length))
    
    # Convert random input vector into fake images
    gen_imgs = generator.predict(z)
    
    # Rescale the pixel values into [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(grid_rows, grid_columns, figsize = (4, 4), sharey = True, sharex = True)
    fig.suptitle('MNIST GAN', fontsize = 10)

    # Write each image into the grid of images
    cnt = 0
    for i in range(grid_rows):
        for j in range(grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap = 'gray')
            axs[i, j].axis('off')
            cnt += 1

##########################################################
# Create the Generator network
generator = Sequential()
generator.add(Dense(256 * 7 * 7, input_dim = noise_vector_length))
generator.add(Reshape((7, 7, 256)))
generator.add(Conv2DTranspose(128, kernel_size = 3, strides = 2, padding = 'same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(alpha = 0.01))
generator.add(Conv2DTranspose(64, kernel_size = 3, strides = 1, padding = 'same'))
generator.add(BatchNormalization()) # Add Batch normalization
generator.add(LeakyReLU(alpha = 0.01)) 
generator.add(Conv2DTranspose(1, kernel_size = 3, strides = 2, padding = 'same'))
generator.add(Activation('tanh'))


# Create the discriminator network
discriminator = Sequential()
discriminator.add( Conv2D(32, kernel_size = 3, strides = 2, input_shape = img_shape, padding='same'))
discriminator.add(LeakyReLU(alpha = 0.01))
discriminator.add( Conv2D(64, kernel_size = 3, strides = 2, input_shape = img_shape, padding = 'same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(alpha = 0.01))
discriminator.add( Conv2D(128, kernel_size = 3, strides = 2, input_shape = img_shape, padding = 'same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(alpha = 0.01))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation = 'sigmoid'))

##########################################################


###########################################################
# Compile the two models which constitute the GAN
discriminator.compile(loss = 'binary_crossentropy', optimizer = Adam(), metrics = ['accuracy'])

# These keeps the weights from changing when we update the weights for the generator
discriminator.trainable = False


##########################################################
# Build the GAN
gan = Sequential()
gan.add(generator)
gan.add(discriminator)

gan.compile(loss = 'binary_crossentropy', optimizer=Adam() )

batchSize = 30
numEpochs = 1 # 10000

# Train
for i in range(numEpochs):

    (X_train, _), (_, _) = mnist.load_data()
    
    # Rescale the input images
    X_train = X_train / 127.5 - 1.0

    # Reshape images
    X_train = np.expand_dims(X_train, axis = 3)
    
    # Labels for the real and fake images
    real = np.ones((batchSize, 1))
    fake = np.zeros((batchSize, 1))

    # Get random index to get a random batch of images
    randomIndex = np.random.randint(0, X_train.shape[0], batchSize)
    imgs = X_train[randomIndex]
    
    # Get a set of random noise vectors
    z = np.random.normal(0, 1, (batchSize, 100))
    # Generate set of fake images
    gen_imgs = generator.predict(z)    
    
    # Train the discriminator on the real images
    d_loss_real = discriminator.train_on_batch(imgs, real)
    # Train the discriminator on the fake images 
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    # The discriminator loss is the average of the two input sets
    d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Generate more random input vectors 
    z = np.random.normal(0, 1, (batchSize, 100))
    
    # Generate more fake images to train with
    #gen_imgs = generator.predict(z)
    
    # Train the generator on the new, fake images 
    # You use the real label because you want the generator's
    # weights to change so as to make the input (z) be classified
    # as a real image
    g_loss = gan.train_on_batch(z, real)

    if ( i % 100 == 0 ):
        
        next_evolution = generator.predict(evolution_vector) 
        
        # Rescale the pixel values into [0, 1]
        next_evolution = 0.5 * next_evolution + 0.5
        
        # Reshape the image into a 28 x 28 array
        next_evolution = np.reshape(next_evolution, (28, 28))
    
        # Convert the image entries in [0, 1] into [0, 255]
        next_evolution = next_evolution * 255

        # Convert from numpy array to an image format
        img = Image.fromarray(next_evolution)
        
        img = img.convert("L") 
        img.save( "evolution/" + str(imageIndex) + '.jpeg')
        imageIndex = imageIndex + 1


# See what the generators learned
generate_images(generator)
plt.show()
generateVideo()

# Save the two networks
gan.save('/home/peter/Desktop/GAN/models/gan.h5') 
discriminator.save('/home/peter/Desktop/GAN/models/discriminator.h5')


