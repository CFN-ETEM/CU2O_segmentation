from scipy.ndimage import rotate
from keras.models import *
from keras.layers import *
from keras.initializers import *
import tensorflow as tf
import numpy as np
from keras.optimizers import *

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

def rotate_image_2d(image, angle):
    """
    Rotates a 2D image clockwise by a given angle and fills blank spaces with a specified value.
    
    Args:
        image (numpy.ndarray): Input image of size (512, 512, 1).
        angle (float): Angle by which to rotate the image clockwise, in degrees.
        fill_value (float): Value to fill blank spaces with.
    
    Returns:
        numpy.ndarray: Rotated image of size (512, 512, 1).
    """
    if image.shape != (512, 512, 1):
        raise ValueError("Input image must have shape (512, 512, 1).")
    fill_value=np.mean(image[0:5,0:5,0])
    # Remove the last dimension to perform 2D rotation
    image_2d = np.squeeze(image, axis=-1)
    
    # Rotate the image clockwise (-angle) with constant mode and custom fill value
    rotated_image = rotate(image_2d, angle=-angle, reshape=False, mode='constant', cval=fill_value)
    
    # Add the last dimension back to maintain shape (512, 512, 1)
    rotated_image = rotated_image[..., np.newaxis]
    
    return rotated_image

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


def changeimage(img,label,max_cval,size):
    img=img/max_cval
    label=label/max_cval
    label[label>0.5]=1
    label[label<=0.5]=0
    imgsize = len(img)
    cutoffx = (imgsize-size)//2
    imgnow=img[cutoffx:imgsize-cutoffx,cutoffx:imgsize-cutoffx]
    labnow=label[cutoffx:imgsize-cutoffx,cutoffx:imgsize-cutoffx]
    return imgnow,labnow

def calculate_metrics(y_true, y_pred):
    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))
    iou = intersection / union if union > 0 else 0.0
    
    tp = np.sum(np.logical_and(y_true, y_pred))
    fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    fn = np.sum(np.logical_and(y_true, np.logical_not(y_pred)))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall, iou

def calculate_threshold_iou(pres, vers, threshold=0.5):
    y_pred = pres.astype(bool)
    y_true = vers.astype(bool)
    
    intersection = np.logical_and(y_pred, y_true)
    union = np.logical_or(y_pred, y_true)
    
    if np.sum(union) == 0:
        return None
    
    iou = np.sum(intersection)/np.sum(union)
    
    if iou >= threshold:
        return iou
    else:
        return None