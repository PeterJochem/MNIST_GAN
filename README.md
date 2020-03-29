# Description
I implemented a generative adversarial network (GAN) trained on the MNIST handwritten digits dataset. I used Keras and Tensorflow to implement the genrator and discriminator networks.

# Results 
The results are pretty stunning! They will not fool a human but they do clearly have some resemblance to the dataset's handwritten digits. 
I only used networks with 2-3 hidden layers! A deper network should deliver more convinving results. Using two convolutional networks should also help too.

![MNIST GAN - Simple Multi Layer Perceptrons](https://github.com/PeterJochem/MNIST_GAN/blob/master/simpleNetworkResults.png  "MNIST GAN - Simple Multi Layer Perceptrons")

This is a video of the generator evolving. Before starting to train the network, I create and store a random input vector for the generator. Every 100 training cycles, I forward prop this vector through the generator and store the resulting image. This is a video of all those images
[![Generator Evolving](https://youtu.be/K0t_Qji7sWk)](https://youtu.be/K0t_Qji7sWk)

# Tensorflow and Virtual Enviroment Setup
It is easiest to run Tensorflow from a virtual enviroment on Linux. Here are instructions on how to setup Tensorflow and the virtual enviroment https://linuxize.com/post/how-to-install-tensorflow-on-ubuntu-18-04/

To activate the virtual enviroment: ```source venv/bin/activate```

