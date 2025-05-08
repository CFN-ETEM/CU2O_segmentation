import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras import backend as keras
from keras.initializers import *
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import csv
from unetmodule import get_unet_with_batchnorm,CustomModelCheckpoint
import sys



tf.keras.backend.clear_session()
print("tf",tf.config.list_physical_devices('GPU'))
ndat=int(sys.argv[1])
exp=int(sys.argv[2])
gpus=tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
data_path=f'./data'
prediction_path=f'./results'
train_data=np.load(f'{data_path}/train.npy')
label_data=np.load(f'{data_path}/label.npy')

model = get_unet_with_batchnorm()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
save_path = f'{prediction_path}/'+'model_{epoch}.hdf5'
checkpoint_callback = CustomModelCheckpoint(save_path=save_path, save_every=10)
csv_logger = CSVLogger(f'{prediction_path}/log.csv', append=False, separator=',')
model.summary()
model.fit(train_data, label_data, batch_size=4, epochs=300, verbose=1, callbacks=[checkpoint_callback,csv_logger])
