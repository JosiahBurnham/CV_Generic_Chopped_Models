import numpy as np
import tensorflow as tf
from tensorflow import keras



from AlexNet.AlexNet import AlexNet
from VGG16.VGG16_Model import VGG_16
from VGG19.VGG19_Model import VGG_19




def main():

    anetData = np.load("AlexNet\\AlexNet_WD\\AlexNet_WD.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    vgg16Data = np.load("VGG16\\VGG_16_WD\\VGG_16_Weights.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    vgg19Data = np.load("VGG19\\VGG_19_WD\\VGG_19_Weights.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    
    vgg16 = VGG_16(vgg16Data, 2, output=False)
    vgg19 = VGG_19(vgg19Data, 2, output=False)
    anet = AlexNet(anetData, 2, output=False)

    model1 = vgg16.get_model(1)
    model2 = vgg19.get_model(1)
    model3 = anet.get_model(1)

    model1.build(( 40,40,1))
    model1.summary()

    print("\n ################################ \n")

    model2.build(( 40,40,1))
    model2.summary()

    print("\n ################################ \n")

    model3.build(( 40,40,1))
    model3.summary()


main()


