import numpy as np
import pandas as pd
from helper import *
from generator import *
from discriminator import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class cGAN():

    def __init__(self, img_shape = [256, 256, 1], gModel = None, dModel = None, ganModel = None):

        self.dModel = dModel
        self.gModel = gModel
        self.ganModel = ganModel
        self.img_shape = img_shape

    def build(self, lr = 0.002, b = [0.5, 0.0], dropG = 0.5, dropD = .5, dModel = None, gModel = None):

        if dModel == None:
	        self.dModel = build_discriminator(self.img_shape, lr, b, dropD)

        if gModel == None:
            self.gModel = build_generator(self.img_shape, dropG)

        self.dModel.trainable = False
        input_img = Input(self.img_shape)
        gen_out = self.gModel(input_img)
        dis_out = self.dModel([input_img, gen_out])
        self.ganModel = Model(input_img, [dis_out, gen_out])

        opt = Adam(learning_rate = lr, beta_1 = b[0])
        self.ganModel.compile(loss = ['binary_crossentropy', 'mae'], optimizer = opt, loss_weights = [1, 100], metrics = [dice_coef])

    def train(self, dataset, n_epochs = 10, n_batch = 1, n_patch = 16):

        def generate_real_samples(dataset, n_samples, patch_shape):
            trainA, trainB = dataset
            ix = np.random.randint(0, trainA.shape[0], n_samples)
            x1, x2 = trainA[ix], trainB[ix]
            y = np.random.uniform(.7, 1.2, (n_samples, patch_shape, patch_shape, 1))
            return [x1, x2], y

        def generate_fake_samples(samples, patch_shape):
            x = self.gModel.predict(samples)
            y = np.random.uniform(0, .3, (len(x), patch_shape, patch_shape, 1))
            return x, y

        trainA, trainB = dataset
        bat_per_epo = int(len(trainA) / n_batch)
        n_steps = bat_per_epo * n_epochs
        dModelHist = []
        gModelHist = []
        gModelHistDSC = []

        for i in range(n_steps):
            [xRealA, xRealB], yReal = generate_real_samples(dataset, n_batch, n_patch)
            xFakeB, yFake = generate_fake_samples(xRealA, n_patch)
            d_loss1 = self.dModel.train_on_batch([xRealA, xRealB], yReal)
            d_loss2 = self.dModel.train_on_batch([xRealA, xFakeB], yFake)
            d_loss = 0.5 * (d_loss1 + d_loss2)
            g_loss = self.ganModel.train_on_batch(xRealA, [yReal, xRealB])

            #if np.mod(i, bat_per_epo):
            print('>%d/%d, d1[%.4f] d2[%.4f] d[%.4f] g[%.4f] DSC[%.4f]' % (i+1, n_steps, d_loss1, d_loss2, d_loss, g_loss[0], g_loss[3]))
            dModelHist.append(d_loss)
            gModelHist.append(g_loss[0])
            gModelHistDSC.append(g_loss[3])

        df = pd.DataFrame(list(zip(gModelHist, dModelHist, gModelHistDSC)), columns = ['gLoss', 'dLoss', 'DSC'])
        df.to_csv('hist.csv')

    def save_weights(self, path = ''):

        self.dModel.save(path + 'dModel.h5')
        self.gModel.save(path + 'gModel.h5')

if __name__ == '__main__':

    model = cGAN()
    model.build()
    model.ganModel.summary()
