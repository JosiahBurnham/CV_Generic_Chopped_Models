#==========================================================
# File   :   AlexNet.py
# Author :   J.Burnham
# Date   :   01/31/2022
# Purpose:  Implementation of the AlexNet Convolutional Neural Network into Keras
#==========================================================

from tensorflow import keras
from tensorflow.keras import layers

class AlexNet(): 
    """ A implementation of the AlexNet Model, where the caller can specify what block
        of the model is desired.
    """
    def __init__(self, wd, num_outputs):
        """property constructor for AlexNet class

        Args:
            wd (array[numpy arrays]): weight array of numpy arrays
            num_outputs (int): number of perdiction classes
        """

        self.wd = wd
        self.num_outputs = num_outputs

        # Conv Layers
        #----------------------------------------
        self.conv1 = layers.Conv2D(filters=64, kernel_size=(11,11), strides=(4,4), padding='same', activation="relu", name="conv1")
        self.conv2 = layers.Conv2D(filters=192, kernel_size=(5,5), strides=(1,1), padding='same', activation="relu", name="conv2")
        self.conv3 = layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv3")
        self.conv4 = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv4")
        self.conv5 = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv5")

        # Fully Connected Layers
        #----------------------------------------
        self.hidden_dense1 = layers.Dense(4096, activation="relu", name="hidden1")
        self.hidden_dense2 = layers.Dense(4096, activation="relu", name="hidden2")
        self.output_layer = layers.Dense(num_outputs,  activation="softmax", name="output")

        # Intermediary Layers
        #----------------------------------------
        self.maxPool = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', data_format="channels_last")
        self.flatten = layers.Flatten()



    def get_model(self, layer_level):
        """get AlexNet varient with only specified blocks

        Args:
            layer_level (int): the block layer that is desired (inclusive)

        Returns:
            keras.model: a VGG_19 model that only contains the specified blocks
        """
        self.__set_layer_weights()

        model = keras.Sequential([])

        if(layer_level >= 1): # block 1
            model.add(layers.Input((224,224,3)))
            model.add(self.conv1)
        if(layer_level >= 2):
            pass #replace with normalization later
        if(layer_level >= 3):
            model.add(self.maxPool)

        if(layer_level >= 4): # block 2
            model.add(self.conv2)
        if(layer_level >= 5):
            pass #replace with normalization later
        if(layer_level >= 6):
            model.add(self.maxPool)

        if(layer_level >= 7): # block 3
            model.add(self.conv3)
        if(layer_level >= 8):
            pass #replace with normalization later
        if(layer_level >= 9):
            model.add(self.conv4)
        if(layer_level >= 10):
            pass #replace with normalization later
        if(layer_level >= 11):
            model.add(self.conv5)
        if(layer_level >= 12):
            pass #replace with normalization later
        if(layer_level >= 13):
            model.add(self.maxPool)      

        if(layer_level >= 14): # block 4  
            model.add(self.flatten)
        if(layer_level >= 15):
            model.add(self.hidden_dense1)
        if(layer_level >= 16):
            model.add(layers.Dropout(0.5))
        if(layer_level >= 17):
            model.add(self.hidden_dense2)
        if(layer_level >= 18):
            model.add(layers.Dropout(0.5))


        if(layer_level < 14):
            model.add(self.flatten)
            
        model.add(self.output_layer)
        return model


        
    def __set_layer_weights(self):
        """set the weights of all the layers in the model, from the weigths that were passed
            at instantiation
        """
        
        # Convolution Layers
        #----------------------------------------
        # Build The Layers
        self.conv1.build((224,224,3))
        self.conv2.build((1,55,55,64))
        self.conv3.build((1,27,27,192))
        self.conv4.build((1,13,13,384))
        self.conv5.build((1,13,13,256))



        # Set the weights
        self.conv1.set_weights([self.wd[0], self.wd[1]])
        self.conv2.set_weights([self.wd[2], self.wd[3]])
        self.conv3.set_weights([self.wd[4], self.wd[5]])
        self.conv4.set_weights([self.wd[6], self.wd[7]])
        self.conv5.set_weights([self.wd[8], self.wd[9]])

        # Fully Connected Layers
        #---------------------------------------
        #Build The Layers
        self.hidden_dense1.build((1,9216))
        self.hidden_dense2.build((1,4096))

        # Set the weights
        self.hidden_dense1.set_weights([self.wd[10], self.wd[11]])
        self.hidden_dense2.set_weights([self.wd[12], self.wd[13]])


    

