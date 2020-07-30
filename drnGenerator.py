from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Concatenate, Dropout, BatchNormalization, Add, Activation, LeakyReLU
from tensorflow.keras.initializers import he_normal

def residual_block(r, f):

    r_short = r

    r = Conv2D(f, (3,3), padding = 'same', kernel_initializer = he_normal())(r)
    r = BatchNormalization()(r)
    r = LeakyReLU(alpha = 0.2)(r)
    r = Conv2D(f, (3,3), padding = 'same', kernel_initializer = he_normal())(r)
    r = BatchNormalization()(r)

    r = Add()([r_short, r])
    r = LeakyReLU(alpha = 0.2)(r)

    return r

def build_generator(img_shape = [256, 256, 1], drop = 0.5):

    input_img = Input(shape = img_shape)

    init = he_normal()

    # Encoder
    g1 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(input_img)
    g1 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(g1)
    g1 = BatchNormalization()(g1)
    g1 = Dropout(drop)(g1, training = True)
    g1 = LeakyReLU(alpha = 0.2)(g1)

    g2 = Conv2D(128, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g1)
    g2 = Conv2D(128, (3,3), padding = 'same', kernel_initializer = init)(g2)
    g2 = BatchNormalization()(g2)
    g2 = Dropout(drop)(g2, training = True)
    g2 = LeakyReLU(alpha = 0.2)(g2)

    g3 = Conv2D(256, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g2)
    g3 = Conv2D(256, (3,3), padding = 'same', kernel_initializer = init)(g3)
    g3 = BatchNormalization()(g3)
    g3 = Dropout(drop)(g3, training = True)
    g3 = LeakyReLU(alpha = 0.2)(g3)

    g4 = Conv2D(512, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g3)
    g4 = Conv2D(512, (3,3), padding = 'same', kernel_initializer = init)(g4)
    g4 = BatchNormalization()(g4)
    g4 = Dropout(drop)(g4, training = True)
    g4 = LeakyReLU(alpha = 0.2)(g4)

    # Bottleneck
    g = Conv2D(1024, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g4)
    g = Conv2D(1024, (3,3), padding = 'same', kernel_initializer = init)(g)
    g = BatchNormalization()(g)
    g = Dropout(drop)(g, training = True)
    g = LeakyReLU(alpha = 0.2)(g)

    # Decoder
    g5 = Conv2DTranspose(512, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g)
    g5 = Add()([g4, g5])
    g5 = Conv2D(256, (3,3), padding = 'same', kernel_initializer = init)(g5)
    g5 = Conv2D(256, (3,3), padding = 'same', kernel_initializer = init)(g5)
    g5 = BatchNormalization()(g5)
    g5 = Dropout(drop)(g5, training = True)
    g5 = LeakyReLU(alpha = 0.2)(g5)

    g6 = Conv2DTranspose(256, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g5)
    g6 = Add()([g3, g6])
    g6 = Conv2D(128, (3,3), padding = 'same', kernel_initializer = init)(g6)
    g6 = Conv2D(128, (3,3), padding = 'same', kernel_initializer = init)(g6)
    g6 = BatchNormalization()(g6)
    g6 = Dropout(drop)(g6, training = True)
    g6 = LeakyReLU(alpha = 0.2)(g6)

    g7 = Conv2DTranspose(128, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g6)
    g7 = Add()([g2, g7])
    g7 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(g7)
    g7 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(g7)
    g7 = BatchNormalization()(g7)
    g7 = Dropout(drop)(g7, training = True)
    g7 = LeakyReLU(alpha = 0.2)(g7)

    g8 = Conv2DTranspose(64, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g7)
    g8 = Add()([g1, g8])
    g8 = Conv2D(1, (3,3), padding = 'same', kernel_initializer = init)(g8)
    g8 = Conv2D(1, (3,3), padding = 'same', kernel_initializer = init)(g8)
    g8 = BatchNormalization()(g8)
    g8 = Dropout(drop)(g8, training = True)
    g8 = Activation('tanh')(g8)

    g0 = residual_block(g8, 1)
    g0 = Add()([g8, g0])

    g1_ = residual_block(g0, 64)
    g1_ = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(g1_)
    g1_ = BatchNormalization()(g1_)
    g1_ = Dropout(drop)(g1_, training = True)
    g1_ = LeakyReLU(alpha = 0.2)(g1_)

    g2_ = Conv2D(64, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g1_)
    g2_ = Add()([g7, g2_])
    g2_ = residual_block(g2_, 64)
    g2_ = Conv2D(128, (3,3), padding = 'same', kernel_initializer = init)(g2_)
    g2_ = BatchNormalization()(g2_)
    g2_ = Dropout(drop)(g2_, training = True)
    g2_ = LeakyReLU(alpha = 0.2)(g2_)

    g3_ = Conv2D(128, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g2_)
    g3_ = Add()([g6, g3_])
    g3_ = residual_block(g3_, 128)
    g3_ = Conv2D(256, (3,3), padding = 'same', kernel_initializer = init)(g3_)
    g3_ = BatchNormalization()(g3_)
    g3_ = Dropout(drop)(g3_, training = True)
    g3_ = LeakyReLU(alpha = 0.2)(g3_)

    g4_ = Conv2D(256, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g3_)
    g4_ = Add()([g5, g4_])
    g4_ = residual_block(g4_, 256)
    g4_ = Conv2D(512, (3,3), padding = 'same', kernel_initializer = init)(g4_)
    g4_ = BatchNormalization()(g4_)
    g4_ = Dropout(drop)(g4_, training = True)
    g4_ = LeakyReLU(alpha = 0.2)(g4_)

    g_ = Conv2D(512, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g4_)
    g_ = residual_block(g_, 512)
    g_ = Conv2D(1024, (3,3), padding = 'same', kernel_initializer = init)(g_)
    g_ = BatchNormalization()(g_)
    g_ = Dropout(drop)(g_, training = True)
    g_ = LeakyReLU(alpha = 0.2)(g_)

    g5_ = Conv2DTranspose(512, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g_)
    g5_ = Add()([g4_, g5_])
    g5_ = residual_block(g5_, 512)
    g5_ = Conv2D(256, (3,3), padding = 'same', kernel_initializer = init)(g5_)
    g5_ = BatchNormalization()(g5_)
    g5_ = Dropout(drop)(g5_, training = True)
    g5_ = LeakyReLU(alpha = 0.2)(g5_)

    g6_ = Conv2DTranspose(256, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g5_)
    g6_ = Add()([g3_, g6_])
    g6_ = residual_block(g6_, 256)
    g6_ = Conv2D(128, (3,3), padding = 'same', kernel_initializer = init)(g6_)
    g6_ = BatchNormalization()(g6_)
    g6_ = Dropout(drop)(g6_, training = True)
    g6_ = LeakyReLU(alpha = 0.2)(g6_)

    g7_ = Conv2DTranspose(128, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g6_)
    g7_ = Add()([g2_, g7_])
    g7_ = residual_block(g7_, 128)
    g7_ = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(g7_)
    g7_ = BatchNormalization()(g7_)
    g7_ = Dropout(drop)(g7_, training = True)
    g7_ = LeakyReLU(alpha = 0.2)(g7_)

    g8_ = Conv2DTranspose(64, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g7_)
    g8_ = Add()([g1_, g8_])
    g8_ = residual_block(g8_, 64)
    g8_ = Conv2D(1, (3,3), padding = 'same', kernel_initializer = init)(g8_)
    g8_ = BatchNormalization()(g8_)
    g8_ = Dropout(drop)(g8_, training = True)
    g8_ = Activation('tanh')(g8_)

    model = Model(input_img, g8_)

    return model


if __name__ == '__main__':

    model = build_generator()
    model.summary()
