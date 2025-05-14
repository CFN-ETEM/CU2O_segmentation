import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from unetmodule import get_unet_with_batchnorm, CustomModelCheckpoint

tf.keras.backend.clear_session()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus, 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Enabled memory growth for {len(gpus)} GPU(s)")
else:
    print("No GPU devices found.")
data_path=f'./data'
prediction_path=f'./results'
os.makedirs(prediction_path, exist_ok=True)

train_data=np.load(f'{data_path}/train.npy')
label_data=np.load(f'{data_path}/label.npy')
print('shapes',train_data.shape,label_data.shape)
model = get_unet_with_batchnorm()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
save_path = f'{prediction_path}/'+'model_{epoch}.hdf5'
checkpoint_callback = CustomModelCheckpoint(save_path=save_path, save_every=10)
csv_logger = CSVLogger(f'{prediction_path}/log.csv', append=False, separator=',')
model.summary()
model.fit(train_data, label_data, batch_size=4, epochs=300, verbose=1, callbacks=[checkpoint_callback,csv_logger])
