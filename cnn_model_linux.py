

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:30:46 2022

@author: bckandemir2018
"""


from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import os.path

from tensorflow.keras.callbacks import ModelCheckpoint,\
                        EarlyStopping, CSVLogger
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout,Flatten,\
                         Conv2D, MaxPooling2D, ZeroPadding2D,\
                         Input, GlobalAveragePooling2D,\
                         TimeDistributed,LSTM,\
                         AveragePooling2D, Concatenate, BatchNormalization
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import backend as K
import imgaug as ia
import imgaug.augmenters as iaa
import os   
import sys
import shutil
import csv
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#set parameters
BATCH_SIZE = 64
AUGMENT = False
# N_CHANNEL = 3 #rgb
N_CHANNEL = 3 #with ycbcr mask

cwd = os.getcwd()

DATASETS_PATH = os.path.join(cwd, "datasets7")
DATA_PATH = os.path.join(DATASETS_PATH, "met_data7")



print(sys.platform)
if sys.platform == "win32":
    
    PATH_DELIM="\\"
    USE_MULTIP = False
    
elif sys.platform == "linux": 
    
    PATH_DELIM = "/"
    USE_MULTIP = True


SEED_VAL = 340
np.random.seed(SEED_VAL) 
ia.seed(SEED_VAL)


tstamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Helper: Save the model.
check_dir = 'checkpoints_' +  tstamp
os.mkdir(check_dir)
checkpointer = ModelCheckpoint(
    filepath=os.path.join(check_dir, 'm' +datetime.datetime.now().strftime("%m%d")\
                          +'.{epoch:03d}-{val_loss:.2f}.hdf5'),\
                         monitor='val_loss', verbose=1, \
                        save_best_only=True, mode='auto')

# Helper: TensorBoard
log_dir = os.path.join("logs7", "fit", tstamp)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

csv_logger = CSVLogger("training_" + tstamp + ".log")

############################################################################

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=50)
def add_denseLayer(inp, growth_rate):#bottleneck layer
    outp = Conv2D(filters=3*growth_rate, kernel_size=1, \
                  strides=1, use_bias=False)(inp)
    outp = BatchNormalization(epsilon=1.001e-5)(outp)
    outp = Activation('relu')(outp)
    #
    outp = Conv2D(filters=growth_rate, kernel_size=3, \
                  strides=1, padding='same', use_bias=False)(outp)
    outp = BatchNormalization(epsilon=1.001e-5)(outp)
    outp = Activation('relu')(outp)
    #
    outp = Concatenate()([inp,outp])
    
    return outp;

def add_denseBlock(inp, numOfLayer, growth_rate):
    
    if numOfLayer < 1:
        return inp;
    
    outp = add_denseLayer(inp, growth_rate)
    for i in range(numOfLayer-1):
        outp = add_denseLayer(outp, growth_rate)
    
    return outp;

def add_transitionBlock(inp, theta):
    #theta*m,size. determines reduction
    prev = backend.int_shape(inp)[-1]
    outp = Conv2D(filters=int(prev*theta), kernel_size=1, padding='same', \
                  strides=1, use_bias=False)(inp)
    outp = BatchNormalization(epsilon=1.001e-5)(outp)
    outp = Activation('relu')(outp)
    
    outp = AveragePooling2D(pool_size=2, strides=2)(outp)
    
    return outp;

def build_denseNet(input_shape, numOfCatg, opt, growth_rate=32, theta=0.5):
    inp = Input(shape=input_shape)
    #default channels_last
    pre = ZeroPadding2D(padding=3)(inp)
    ####
    pre = Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False)(pre)
    pre = BatchNormalization(epsilon=1.001e-5)(pre)
    pre = Activation('relu')(pre)
    ####
    pre = ZeroPadding2D(padding=1)(pre)
    pre = MaxPooling2D(pool_size=3, strides=2)(pre)

    numOfLayer = [2,3,4,3]

    first = add_denseBlock(pre, numOfLayer[0], growth_rate)
    first = add_transitionBlock(first, theta)
    #2
    second = add_denseBlock(first, numOfLayer[1], growth_rate)
    second = add_transitionBlock(second, theta)
    #3
    third = add_denseBlock(second, numOfLayer[2], growth_rate)
    third = add_transitionBlock(third, theta)
    #4
    fourth = add_denseBlock(third, numOfLayer[3], growth_rate)
    ##########################
    last = GlobalAveragePooling2D(name='last_layer')(fourth)
    outp = Dense(numOfCatg, activation='softmax')(last)
    #done
    
    model = Model(inputs=inp, outputs=outp)
    
    model.summary()
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, \
    #              metrics=['accuracy'])
                  metrics=['accuracy',f1_m, precision_m, recall_m])
    
    return model;
############################################################################


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def getClasses(path = "classIndex5.txt"):
    
    with open(path) as fin:
        classes = [row.strip() for row in list(fin)]
    
    return classes
############################################################################


class denseNet_generator(Sequence):
    
    def __init__(self, paths, batch_size, img_size,
                  n_channel, shuffle, augment=False):
        
        self.paths = paths
        np.random.shuffle(self.paths)
        self.img_size = img_size
        self.n_channel = n_channel
        self.batch_size = batch_size
        self.classes = getClasses()
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
        
        
    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paths) / self.batch_size))
    
    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_paths = [self.paths[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(batch_paths)
        
        #preprocessing and augmentation
        if self.augment:
            X = self.augmentor(X)

        return X, y
    
    
    def on_epoch_end(self):#'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    
    def __data_generation(self, batch_paths):

        X = np.empty((self.batch_size, 
                      *self.img_size, self.n_channel))
        y = np.empty((self.batch_size, len(self.classes)), dtype=int)

        #print("generation X.sh:",X.shape, y.shape)
        # Generate data
        for i, path in enumerate(batch_paths):
            
            # Store sample
            X[i] = self.load_X(path)
            y[i] = self.load_y(path)
            
        return X, y
    

    def load_X(self, path): #rgb + ycbcr_mask channel

        #goruntu bgr + 
        org = Image.open(path)
        
        resized = org.resize(self.img_size)
        
        data = np.asarray(resized)
        if self.n_channel == 4:
            
            ycbcr_img = resized.convert('YCbCr')
            
            ycbcr_data = np.asarray(ycbcr_img)
            
            mask = np.zeros((*self.img_size, 1)).astype(np.uint8)
            
            #133-173 cr 2
            #80-120 cb 1
            subsetter = np.where(((ycbcr_data[:,:,2] <= 173) & (ycbcr_data[:,:,2] >= 133) & \
                    (ycbcr_data[:,:,1] <= 120) & (ycbcr_data[:,:,1] >= 80)))

            mask[subsetter] = 255

            X = np.concatenate((data, mask), axis=2)
        
        elif self.n_channel == 3:
            
            X = data
            
        X = X / 255 #normalization
        #print(X)
        return X

    def load_y(self, path):

        offset = len(DATASETS_PATH.split(PATH_DELIM))
        
        label = self.classes.index(path.split(PATH_DELIM)[offset+2])

        one_hot_y = to_categorical(label,\
                                    num_classes=len(self.classes))
        
        #print("y.sh:", one_hot_y.shape)
        return one_hot_y
    
    
    def augmentor(self, images):
        
        seq = iaa.Sequential([
                
            iaa.Affine(
                scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                translate_percent={"x": (-0.3, 0.3), "y": (-0.2, 0.2)},
                rotate=(-30, 30),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            ),
            #iaa.Multiply((0.2, 1.2))
            ], random_order=True)
        
        
        #normalization /255
        aug_images = seq.augment_images(images)
        #aug_images = aug_images / 255 

        return aug_images



def load_paths(set_name):
    
    if sys.platform == 'win32':
        
        fname = os.path.join(DATA_PATH, set_name + "_paths7.txt")
        
    else:
        
        fname = os.path.join(DATA_PATH, set_name + "_paths_linux7.txt")
        
        
    with open(fname) as fin:
        paths = [os.path.join(DATASETS_PATH,row.strip()) for row in list(fin)]
        
    #print("last pth:", paths[-1])
    return paths

def main():
    
    #in general, generate when dataset generation
    #iterate_files(data_path = "ash_data")
    
    params = {'batch_size': 64,
          'img_size': (224,224),
          'n_channel': N_CHANNEL,
          'shuffle': True}
    
    
    train_paths = load_paths("train")
    val_paths = load_paths("validation")
    
    train_generator = denseNet_generator(train_paths, **params, augment=AUGMENT)
    val_generator = denseNet_generator(val_paths, **params)
    
    inp_shape = (*params["img_size"], params["n_channel"])

    model = build_denseNet(input_shape=inp_shape,\
                           numOfCatg = len(getClasses()),\
                           opt = optimizers.Adam())

    
    callbacks = [checkpointer, tensorboard_callback, csv_logger,early_stopper]
    
    
    if USE_MULTIP:
        print("workers:6")
        model.fit( train_generator,
            validation_data=val_generator,
            epochs=500,
            workers=6, use_multiprocessing=True,
            callbacks=callbacks)
    else:
        
        model.fit( train_generator,
            validation_data=val_generator,
            epochs=500,
            callbacks=callbacks)
        
#####################################################################################

if __name__ == '__main__':
    
    main()
    
#####################################################################################



params = {'batch_size': 64,
          'img_size': (224,224),
          'n_channel': 4,
          'shuffle': True}
    

train_paths = load_paths("train")

train_generator = denseNet_generator(train_paths, **params, augment=AUGMENT)

pth = os.path.join(DATASETS_PATH,"/mnt/ntfs2/banucicek/datasets7/met_data7/train/gozalti/gozalti_2197.jpg")
X = train_generator.load_X(pth)
y = train_generator.load_y(os.path.join(DATASETS_PATH,pth))

print("label_1.resim:", y)
#plt.imshow(X[:,:,3], cmap ='gray')
plt.imshow(X[:,:,:3])

#####################################################################################

params = {'batch_size': 64,
          'img_size': (224,224),
          'n_channel': 4,
          'shuffle': True}
    

train_paths = load_paths("train")

train_generator = denseNet_generator(train_paths, **params, augment=AUGMENT)

pth = os.path.join(DATASETS_PATH,"/mnt/ntfs2/banucicek/datasets7/met_data7/train/gozalti/gozalti_5373.jpg")
X = train_generator.load_X(pth)
y = train_generator.load_y(os.path.join(DATASETS_PATH,pth))

print("label_2.resim:", y)
#plt.imshow(X[:,:,3], cmap ='gray')
plt.imshow(X)

