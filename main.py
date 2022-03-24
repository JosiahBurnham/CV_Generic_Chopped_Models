import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy.io

from Scripts.imbed import ImbedDataset
from Scripts.dataset_KFold import KFoldDataset
from Scripts.prep_dataset import PrepDataset
tf.config.run_functions_eagerly(True)


def load_data(self):
    """load the dataset from a .mat

    Returns:
        numpy_array: the raw data from the .mat file
    """
    raw_data = np.zeros(shape=(0,self.one_dim_image_size))
    for i in range(self.num_files):
        train_str = self.file_path + str(i+1) + ".mat"
        file_data = scipy.io.loadmat(train_str)[self.dict_name]
        raw_data = np.append(raw_data, file_data, axis=0)

    raw_data = np.expand_dims(raw_data, axis=2)
    return raw_data

def main():

    model = keras.Sequential([])
    model.add(ImbedDataset(227, 227))

    model.build((1000,40,40,1))
    model.summary()

    
    train_str           = "Train_Data/SHAD_CSHD_SINE_GRY_40x40_1_"
    (x_train, y_train), (x_test, y_test) = KFoldDataset(train_str, 68, num_folds=2).make_dataset()
    (x_train_fold, y_train_fold), (x_test_fold, y_test_fold) = PrepDataset(x_train, y_train, x_test, y_test, 1).prep_epoch_data()

    y_train_fold = y_train_fold[:,1]
    y_test_fold  = y_test_fold[:,1]

    print(x_train_fold.shape)


main()


