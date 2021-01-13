import numpy as np
from PIL import Image
import cv2
from urllib.request import urlopen

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


IMG_SIZE = 128

def convertImg(url):
    return Image.open(urlopen(url)).convert("RGB")

def processImg(url):

    try:
        img = convertImg(url)
        img = cv2.resize(img, dsize = (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
        return img
    except Exception as e:
        print(e)

def Encoder():
    img = Input(shape = (IMG_SIZE, IMG_SIZE, 3))
    x = Conv2D(16, (3, 3), activation="relu", padding="same")(img)
    x = MaxPool2D((2, 2), padding="same")(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same",
               kernel_regularizer=l2(0.001))(x)
    x = MaxPool2D((2, 2), padding="same")(x)
    x = Conv2D(3, (3, 3), activation="relu", padding="same",
               kernel_regularizer=l2(0.001))(x)
    out = MaxPool2D((2, 2))(x)
    x = Flatten()(out)
    encoded = Dense(8, activation="relu")(x)

    img_out = Model(img, out)
    encoder = Model(img, encoded)

    return img_out, encoder

def imgToVec(url):
    img = processImg(url)
    img = np.expand_dims(img, axis = 0)
    _, encoder = Encoder()

    return encoder.predict(img/255)

if __name__=="__main__":
    # sample image url(imageUrl)
    vec = imgToVec('https://akamai.poxo.com/nandaglobal/nandaglobal.cafe24.com/web/product/tiny/202101/64c10421a8068800cab01977ab0332ca.webp')
