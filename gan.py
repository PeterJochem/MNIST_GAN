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



def sample_images(generator, image_grid_rows = 4, image_grid_columns = 4):
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, noise_vector_length))
    gen_imgs = generator.predict(z)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4, 4), sharey=True, sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1





##########################################################
# Create the Generator network
generator = Sequential()
generator.add(Dense(128, input_dim = noise_vector_length ))
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
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Dense(1, activation='sigmoid'))

##########################################################


###########################################################
# Compile the two models which constitute the GAN
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics = ['accuracy'])

# These keeps the weights from changing when we update the weights for the generator
discriminator.trainable = False


##########################################################
# Build the GAN
gan = Sequential()
gan.add(generator)
gan.add(discriminator)

gan.compile(loss='binary_crossentropy', optimizer=Adam())

batchSize = 5
numEpochs = 1

# Train
for i in range(numEpochs):

    (X_train, _), (_, _) = mnist.load_data()
    
    # Rescale the input images
    X_train = X_train / 127.5 - 1.0

    # What does this do?
    X_train = np.expand_dims(X_train, axis = 3)
    

    real = np.ones((batchSize, 1))
    fake = np.zeros((batchSize, 1))

    idx = np.random.randint(0, X_train.shape[0], batchSize)
    imgs = X_train[idx]

    z = np.random.normal(0, 1, (batchSize, 100))
    gen_imgs = generator.predict(z)    
    
    d_loss_real = discriminator.train_on_batch(imgs, real)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)


    z = np.random.normal(0, 1, (batchSize, 100))
    gen_imgs = generator.predict(z)
    g_loss = gan.train_on_batch(z, real)
    


sample_images(generator)
plt.show()


