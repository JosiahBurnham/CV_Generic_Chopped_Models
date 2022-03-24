"""
File    : imbed.py
Author  : J.Burnham
Date    : 2/21/2022
Purpose : to imbed an greyscale image dataset into a larger rbg one
"""

from tensorflow import keras
import tensorflow as tf
import numpy as np


class ImbedDataset(keras.layers.Layer):
    def __init__(self, desired_width, desired_height):
        super(ImbedDataset, self).__init__()
        self.desired_width = desired_width
        self.desired_height = desired_height

    @tf.function
    def call(self, x):
        x = tf.image.grayscale_to_rgb(x)
        x = tf.image.resize_with_pad(x, self.desired_height,self.desired_width)
        print(x)

        return x
