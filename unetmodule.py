from keras.models import *
from keras.layers import *
from keras.initializers import *
import tensorflow as tf
import numpy as np

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, save_path, save_every):
        super().__init__()
        self.save_path = save_path
        self.save_every = save_every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_every == 0:
            model_path = self.save_path.format(epoch=epoch + 1)
            self.model.save(model_path)
            print(f"Model saved at: {model_path}")

def get_unet_with_batchnorm():
    inputs = Input((512, 512, 1))

    # Encoder block 1: from (512,512,1) to (256,256,64)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # Now shape: (256,256,64)

    # Encoder block 2: from (256,256,64) to (128,128,128)
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # Now shape: (128,128,128)

    # Bottleneck (can be seen as the third UNet layer)
    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    # Decoder block 1: upsample from (128,128,128) to (256,256,128) and merge with conv2
    up2 = UpSampling2D(size=(2, 2))(conv3)
    up2 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    merge2 = concatenate([conv2, up2], axis=3)
    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge2)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    # Decoder block 2: upsample from (256,256,128) to (512,512,128) and merge with conv1
    up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    merge1 = concatenate([conv1, up1], axis=3)
    conv5 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge1)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    # Final output layer
    conv6 = Conv2D(1, 1, activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=conv6)
    return model
