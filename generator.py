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

def build_generator(img_shape = [256, 256, 1], drop = 0.5):

    # Conv2D - BatchNormalization - LeakyReLU
    # Conv2DTranspose - BatchNormalization - Dropout - relu
    init = RandomNormal(stddev = 0.02)
    input_img = Input(shape = img_shape)

    # Encoder
    g1 = Conv2D(64, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(input_img)
    g1 = LeakyReLU(alpha = 0.2)(g1)

    g2 = Conv2D(128, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g1)
    g2 = BatchNormalization()(g2, training = True)
    g2 = LeakyReLU(alpha = 0.2)(g2)

    g3 = Conv2D(256, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g2)
    g3 = BatchNormalization()(g3, training = True)
    g3 = LeakyReLU(alpha = 0.2)(g3)

    g4 = Conv2D(512, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g3)
    g4 = BatchNormalization()(g4, training = True)
    g4 = LeakyReLU(alpha = 0.2)(g4)

    g5 = Conv2D(512, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g4)
    g5 = BatchNormalization()(g5, training = True)
    g5 = LeakyReLU(alpha = 0.2)(g5)

    g6 = Conv2D(512, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g5)
    g6 = BatchNormalization()(g6, training = True)
    g6 = LeakyReLU(alpha = 0.2)(g6)

    g7 = Conv2D(512, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g6)
    g7 = BatchNormalization()(g7, training = True)
    g7 = LeakyReLU(alpha = 0.2)(g7)

    # Bottleneck
    g = Conv2D(512, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g7)
    g = Activation('relu')(g)

    # Decoder
    g = Conv2DTranspose(512, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g)
    g = BatchNormalization()(g, training = True)
    g = Dropout(drop)(g, training = True)
    g = Concatenate()([g7, g])
    g = Activation('relu')(g)

    g = Conv2DTranspose(512, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g)
    g = BatchNormalization()(g, training = True)
    g = Dropout(drop)(g, training = True)
    g = Concatenate()([g6, g])
    g = Activation('relu')(g)

    g = Conv2DTranspose(512, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g)
    g = BatchNormalization()(g, training = True)
    g = Dropout(drop)(g, training = True)
    g = Concatenate()([g5, g])
    g = Activation('relu')(g)

    g = Conv2DTranspose(512, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g)
    g = BatchNormalization()(g, training = True)
    g = Dropout(drop)(g, training = True)
    g = Concatenate()([g4, g])
    g = Activation('relu')(g)

    g = Conv2DTranspose(256, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g)
    g = BatchNormalization()(g, training = True)
    g = Dropout(drop)(g, training = True)
    g = Concatenate()([g3, g])
    g = Activation('relu')(g)

    g = Conv2DTranspose(128, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g)
    g = BatchNormalization()(g, training = True)
    g = Dropout(drop)(g, training = True)
    g = Concatenate()([g2, g])
    g = Activation('relu')(g)

    g = Conv2DTranspose(64, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g)
    g = BatchNormalization()(g, training = True)
    g = Dropout(drop)(g, training = True)
    g = Concatenate()([g1, g])
    g = Activation('relu')(g)

    g = Conv2DTranspose(1, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(g)
    out = Activation('tanh')(g)

    model = Model(input_img, out)

    return model

if __name__ == '__main__':

    model = build_generator()
    model.summary()
