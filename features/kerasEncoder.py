import numpy as np
from PIL import Image
import cv2
from urllib.request import urlopen
from io import BytesIO

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from itertools import chain


IMG_SIZE = 128

def convertImg(url):
    img = Image.open(urlopen(url)).convert("RGB")
    # save temporarily
    temp_byte = BytesIO()
    img.save(temp_byte, "jpeg")
    img = np.array(Image.open(temp_byte).convert("RGB"))
    return img

def processImg(url):
    try:
        img = cv2.cvtColor(convertImg(url), cv2.COLOR_BGR2RGB)
        # PIL은 RGB 순서대로 표기하고, cv2는 BGR 순서대로 표기함.
        # 시각화 하려면 다시 RGB로 변환하는 프로세스 추가
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
    print(img.shape)
    _, encoder = Encoder()

    return list(chain.from_iterable(encoder.predict(img/255).tolist()))

if __name__=="__main__":
    # sample image url(imageUrl)
    vec = imgToVec('https://www.stylenanda.com/web/product/tiny/20200403/87c1d5150cc2239d128836f6ec19811f.webp')
    print(vec)