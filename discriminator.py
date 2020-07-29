import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

def build_discriminator(img_shape = [256, 256, 1]):

    init = RandomNormal(stddev = 0.02)
    input_img = Input(shape = img_shape)
    input_tar = Input(shape = img_shape)
    input = Concatenate()([input_img, input_tar])

    d = Conv2D(64, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(input)
    d = LeakyReLU(alpha = 0.2)(d)

    d = Conv2D(128, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha = 0.2)(d)

    d = Conv2D(256, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha = 0.2)(d)

    d = Conv2D(512, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha = 0.2)(d)

    d = Conv2D(512, (4,4), padding = 'same', kernel_initializer = init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha = 0.2)(d)

    d = Conv2D(1, (4,4), padding = 'same', kernel_initializer = init)(d)
    out = Activation('sigmoid')(d)

    model = Model([input_img, input_tar], out)
    opt = Adam(lr = 0.0002, beta_1 = 0.5)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, loss_weights = [0.5])

    return model

if __name__ == '__main__':

    model = build_discriminator()
    model.summary()
