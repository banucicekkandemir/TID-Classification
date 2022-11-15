# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:51:23 2022

@author: bckandemir2018
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:07:00 2022

@author: bckandemir2018
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 23:36:38 2022

@author: bckandemir2018
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 18:30:06 2022

@author: bckandemir2018
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:20:16 2022

@author: bckandemir2018
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:33:34 2022

@author: bckandemir2018
"""

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

import imgaug as ia
import numpy as np
import tensorflow as tf
import random

#reproduciblity
SEED_VAL = 340
random.seed(SEED_VAL)
np.random.seed(SEED_VAL) 
ia.seed(SEED_VAL)
tf.random.set_seed(SEED_VAL)

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,ConvLSTM2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import os.path

from tensorflow.keras.callbacks import ModelCheckpoint,\
                        EarlyStopping, CSVLogger
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout,Flatten,\
                         Conv2D, Conv3D, MaxPooling2D, ZeroPadding2D,\
                         Input, GlobalAveragePooling2D,\
                         TimeDistributed,LSTM,\
                         AveragePooling2D, Concatenate, BatchNormalization
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

from tensorflow.keras import backend as K

import imgaug.augmenters as iaa
import os   
import sys
import shutil
import csv
import datetime

from PIL import Image, ImageOps
import matplotlib.pyplot as plt



BATCH_SIZE = 256
AUGMENT = False
N_CHANNEL = 3 
WINDOW_SIZE = 25 
N_FEATURE = 15

#STRIDE = 3 #get one frame per three ones
#FPS = 30
#N_CHANNEL = 4 #with ycbcr mask

cwd = os.getcwd()

# DATASETS_PATH = os.path.join(cwd, "..", "..", "datasets")
# #DATA_PATH = os.path.join(DATASETS_PATH, "med_data_v")
# IDS_PATH = os.path.join(DATASETS_PATH, "med_FL_" + str(WINDOW_SIZE))

DATASETS_PATH = os.path.join(cwd, "datasets7")
IDS_PATH = os.path.join(DATASETS_PATH, "med_data_fextractor_ids"+ str(WINDOW_SIZE))

#print(sys.platform)
if sys.platform == "win32":
    
    PATH_DELIM = "\\"
    USE_MULTIP = False
    
elif sys.platform == "linux": 
    
    PATH_DELIM = "/"
    USE_MULTIP = True


tstamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Helper: Save the model.
check_dir = 'checkpoints_' +  tstamp
os.mkdir(check_dir)
checkpointer = ModelCheckpoint(
    filepath=os.path.join(check_dir, 'm' +datetime.datetime.now().strftime("%m%d")\
                          +'.{epoch:03d}-{val_loss:.2f}.hdf5'),\
                         monitor='val_loss', verbose=1, \
                        save_best_only=False, mode='auto')

# Helper: TensorBoard
log_dir = os.path.join("logs_lstm_6", "fit", tstamp)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

csv_logger = CSVLogger("training_" + tstamp + ".logs_lstm_6")

###########################################################
def build_lstm_model(input_shape, numOfCatg, opt): #firstmodel
    
    model = Sequential()
    
    #model.add(Masking(mask_value=special_value, input_shape=input_shape))
    
    model.add(LSTM(16, return_sequences=False, dropout=0.3,input_shape=input_shape))
    model.add(BatchNormalization())
    """model.add(LSTM(16, return_sequences=True, dropout=0.3))
    model.add(BatchNormalization())
    model.add(LSTM(8, return_sequences=False, dropout=0.3))
    model.add(BatchNormalization())
    model.add(LSTM(256, return_sequences=True, dropout=0.3,input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LSTM(128, return_sequences=False, dropout=0.3))
    model.add(BatchNormalization())"""
    
    #model.add(TimeDistributed(Dense(64)))
    model.add(Dense(16))
    model.add(BatchNormalization())
    
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    #model.add(TimeDistributed(Dense(numOfCatg)))
    model.add(Dense(numOfCatg))
    model.add(Activation('softmax'))
    
    #print(model.count_params())
    model.summary()
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, \
                  metrics=['accuracy', tf.keras.metrics.AUC(),
                           f1_m, precision_m, recall_m])
    
    return model;

###########################################################
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


###########################################################

class denseNet_generator(Sequence):
    
    def __init__(self, ids, batch_size, window_size,n_feature,
                 shuffle, augment=False):

        
        self.ids = ids
        if shuffle:
            random.shuffle(self.ids)
            
        self.n_feature = n_feature
        self.window_size = window_size
        self.batch_size = batch_size
        self.classes = getClasses()
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
        
        
    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))
    
    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_ids = [self.ids[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(batch_ids)
        
        
        return X, y
    
    
    def on_epoch_end(self):#'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    
    def __data_generation(self, batch_ids):
        #'Generates data containing batch_size samples'
        
        #sample_shape = (224,224,4)
        # X : (n_samples, 4) rgb + ycbcr_MASK
        # Initialization
        X = np.empty((self.batch_size, self.window_size,self.n_feature),
                     dtype= np.float64)
        y = np.empty((self.batch_size, len(self.classes)))

        
        
        # Generate data
        for i, batch_id in enumerate(batch_ids):
            
            # Store sample
            X[i] = self.load_X(batch_id)
            y[i] = self.load_y(batch_id)

            
        #print(X.shape, y.shape) 
        
        #print(X.shape, y.shape, w.shape)
        return X, y
    

    def load_X(self, batch_id): 
        
        frame_paths = batch_id["path"]
        
        #print("path:", frame_paths)
            
        X = np.asarray([])
        
        i = 0
        for f in frame_paths:
            
            fpath = os.path.join(DATASETS_PATH, f)
        
            features = np.expand_dims(np.load(fpath), axis=0)
            
            
            if i == 0:
                X = features
            else:
                X = np.concatenate((X, features), axis=0)

            i += 1
            
        #print(X.shape, X.dtype.name)          
        return X

    def load_y(self, batch_id):

        label = self.classes.index(batch_id["label"])

        one_hot_y = to_categorical(label,\
                                    num_classes=len(self.classes))
        
        #print("y.sh:", one_hot_y.shape)
        return one_hot_y

###########################################################
def load_ids(set_name): #train / val
    
    ids = []
    
    if sys.platform == 'win32':
        
        fname = os.path.join(IDS_PATH, set_name + "_win7.csv")
    else:
        
        fname = os.path.join(IDS_PATH, set_name + "_lin7.csv")
        
        
    with open(fname) as fin:
        paths = np.array([row.strip().split(",") for row in list(fin)])
        
    fname_labels = os.path.join(IDS_PATH, set_name + "_label7.csv")
    with open(fname_labels) as fin:
        labels = np.array([row.strip() for row in list(fin)])

        
    #pathsNlabels = np.vstack((paths,labels)) 
        
    # fname_weights = os.path.join(IDS_PATH, set_name + "_weights.csv")
    # with open(fname_weights) as fin:
        
    #     weights = [row.strip().split(",") for row in list(fin)]
        
        
    
    # #print(weights)
    # w_dict = {}
    # for w in weights:
        
    #     w_dict[w[0]] = float(w[1])
        
        
    
    for i in range(len(paths)):
        
        item = {}
        item["path"] = paths[i]
        item["label"] = labels[i]
        # item["weight"] = w_dict[labels[i]]
        
        #print(item)
        ids.append(item)
        
    return ids


###########################################################

###########################################################
def main():
    
    
    params = {'batch_size': BATCH_SIZE,
      'window_size': WINDOW_SIZE,
      'n_feature': N_FEATURE,
      'shuffle': True}
    
    
    train_ids = load_ids("train")
    val_ids = load_ids("validation")
    
    train_generator = denseNet_generator(train_ids, **params, augment=AUGMENT)
    val_generator = denseNet_generator(val_ids, **params)
    
    inp_shape = (params["window_size"], params["n_feature"])

    model = build_lstm_model(input_shape=inp_shape,\
                           numOfCatg = len(getClasses()),\
                           opt = optimizers.Adam(learning_rate=0.001))

    
    callbacks = [checkpointer, tensorboard_callback, csv_logger]
    
    if USE_MULTIP:
        print("workers:6")
        model.fit( train_generator,
            validation_data=val_generator,
            epochs=100,
            workers=6, use_multiprocessing=True,
            callbacks=callbacks)
    else:
        
        model.fit( train_generator,
            validation_data=val_generator,
            epochs=100,
            callbacks=callbacks)
###########################################################

if __name__ == '__main__':
    
    main()