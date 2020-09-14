import cv2
import sklearn
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from os import listdir
from os import system, name
from os.path import isfile, join
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def clear():

    if name == 'nt':
        _ = system('cls')

    else:
        _ = system('clear')

def loadImages(mypath, resize = True, size = 256):

    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    onlyfiles.sort()
    images = np.zeros((len(onlyfiles), size, size))

    for n in range(len(onlyfiles)):

        clear()
        im = cv2.imread(mypath + "/" + onlyfiles[n], 0)
        im = cv2.resize(im, (size, size), interpolation = cv2.INTER_AREA)
        #cv2_imshow(im)
        images[n,:,:] = im

        print("Files loaded: {}/{}".format(n+1, len(onlyfiles)))

    print("Total Images Loaded: " + str(images.shape[0]))

    return images

def loadData(mypath, nFiles = 0, size = 256, channels = 1):

    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    onlyfiles.sort()
    images = np.empty(len(onlyfiles), dtype=object)
    images = np.zeros((len(onlyfiles), size, size))

    if nFiles == 0:
        nFiles = len(onlyfiles)

    for n in range(len(onlyfiles)):

        if n >= nFiles:
            break

        if n == 0:
            images = loadImagesNIB(mypath + onlyfiles[n])
        else:
            images = np.concatenate([images, loadImagesNIB(mypath + onlyfiles[n])])

        print("Files loaded: {}/{}".format(n+1, nFiles))

    print("Total Images Loaded: " + str(images.shape[0]))

    return images

def loadImagesNIB(path, resize = False):

    data = nib.load(path)
    data = data.get_fdata()
    lim = 256
    img = np.zeros([data.shape[2], lim, lim])

    if resize == True:

        for i in range(data.shape[2]):

            img[i] = cv2.resize(data[:,:,i], (256,256), interpolation = cv2.INTER_AREA)

    else:

        for i in range(data.shape[2]):

            img[i] = data[:lim, :lim, i]

    return img

def reshapeImages(raw, label, dataSize = .80):

    images = raw.reshape(raw.shape[0], raw.shape[1], raw.shape[2], 1)
    labels = label.reshape(label.shape[0], label.shape[1], label.shape[2], 1)

    l = (int(images.shape[0]*(1-dataSize))) + 1

    x_train = images[:-l]
    y_train = labels[:-l]
    x_dev = images[-l:-int(l/2)]
    y_dev = labels[-l:-int(l/2)]
    x_test = images[-int(l/2):]
    y_test = labels[-int(l/2):]

    return x_train, y_train, x_test, y_test, x_dev, y_dev

def generateZoom(imgs, labs, samples):

    rawGen = []
    labelGen = []
    n = imgs.shape[0]

    for j in range(n):

        img = imgs[j]
        lab = labs[j]
        seed = np.random.randint(1000)
        sampleimg = np.expand_dims(img, 0)
        samplelab = np.expand_dims(lab, 0)
        datagen = ImageDataGenerator(zoom_range = [0.5, 1.0])
        itimg = datagen.flow(sampleimg, batch_size = 1, seed = seed)
        itlabel = datagen.flow(samplelab, batch_size = 1, seed = seed)

        for i in range(samples):

            batchraw = itimg.next()
            imageraw = batchraw[0].astype('uint8')
            batchlabel = itlabel.next()
            imagelabel = batchlabel[0].astype('uint8')
            rawGen.append(imageraw)
            labelGen.append(imagelabel)

    rawGen = np.array(rawGen)
    labelGen = np.array(labelGen)

    return rawGen, labelGen

def shuffle(raw, label):

	raw_shuffled, label_shuffled = sklearn.utils.shuffle(raw, label)

	return raw_shuffled, label_shuffled

def getFourier(img):

    fourier = []
    for i in range(img.shape[0]):

        temp = img[i]
        f = np.fft.fft2(temp)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        fourier.append(magnitude_spectrum)

    return np.array(fourier)

def histeq(img, size = 256, flag = True):

    hist,bins = np.histogram(img.flatten(), size, [0,size])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    """
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
    """

    if flag:
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')

    img2 = cdf[img]

    return img2


def flipInd(imgOrig):

    img = imgOrig.reshape(imgOrig.shape[0], imgOrig.shape[1], 1)
    data = img_to_array(img)
    samples = np.expand_dims(data, 0)
    datagen = ImageDataGenerator(horizontal_flip = True)
    it = datagen.flow(samples, batch_size = 1)

    while True:
        # define subplot
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')

        if not np.array_equal(data, image):
            break

    return image.reshape(imgOrig.shape)

def genFlip(img, lab):

    imgGen = flipInd(img)
    labGen = flipInd(lab)

    return imgGen, labGen

def flip(raw, lab, save = False, show = False):

    rawGen = np.zeros(raw.shape)
    labGen = np.zeros(lab.shape)

    for i in range(raw.shape[0]):

        rawGen[i], labGen[i] = genFlip(raw[i], lab[i])
        nameR = 'flip/flipRaw' + str(i) + '.jpg'
        nameL = 'flip/flipLab' + str(i) + '.jpg'

        if save:
            cv2.imwrite(nameR, rawGen[i])
            cv2.imwrite(nameL, labGen[i])
        if show:
            cv2.imshow(nameR, rawGen[i])
            cv2.imshow(nameL, labGen[i])
            cv2.waitKey()

    return rawGen, labGen

def zoom(raw, lab, samples, save = False, show = False):

    raw = raw.reshape(raw.shape[0], raw.shape[1], raw.shape[2], 1)
    lab = lab.reshape(lab.shape[0], lab.shape[1], lab.shape[2], 1)
    rawGen, labGen = generateZoom(raw, lab, samples)

    for i in range(rawGen.shape[0]):

        nameR = 'zoom/zoomRaw' + str(i) + '.jpg'
        nameL = 'zoom/zoomLab' + str(i) + '.jpg'

        if save:
            cv2.imwrite(nameR, rawGen[i])
            cv2.imwrite(nameL, labGen[i])
        if show:
            cv2.imshow(nameR, rawGen[i])
            cv2.imshow(nameL, labGen[i])
            cv2.waitKey()

    return rawGen, labGen

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
