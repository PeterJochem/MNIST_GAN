import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

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

# This takes the gan and creates image with it and plots them
# into a grid of 4x4 images
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
generator.add(Dense(128, input_dim = noise_vector_length ))
generator.add(LeakyReLU(alpha = 0.01))

generator.add(Dense(200))
generator.add(LeakyReLU(alpha = 0.01))

# Empirically, tanh tends to produce less blurry images
generator.add(Dense(28 * 28 * 1, activation='tanh') )

# Reformat the output from one vector shape to another
# that the discriminator network can process
generator.add(Reshape(img_shape))
##########################################################

# Create the discriminator network
discriminator = Sequential()
discriminator.add(Flatten(input_shape = img_shape))
discriminator.add(Dense(128))
discriminator.add(LeakyReLU(alpha = 0.01))
discriminator.add(Dense(128))
discriminator.add(LeakyReLU(alpha = 0.01))
discriminator.add(Dense(1, activation ='sigmoid'))

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
numEpochs = 10000

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
    

# See what the generators learned
generate_images(generator)
plt.show()


