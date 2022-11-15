# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 08:52:25 2022

@author: bckandemir2018
"""

SEED_VAL = 340
import numpy as np
import tensorflow as tf
import random

random.seed(SEED_VAL)
np.random.seed(SEED_VAL) 
tf.random.set_seed(SEED_VAL)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import operator
import random
import glob
import os.path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import sys
from PIL import Image, ImageOps
######################################################

BATCH_SIZE = 256
AUGMENT = False
N_CHANNEL = 3 
WINDOW_SIZE = 25
N_FEATURE = 15

cwd = os.getcwd()
######################################################

DATASETS_PATH = os.path.join(cwd, "datasets7")
IDS_PATH = os.path.join(DATASETS_PATH, "med_data_fextractor_ids"+ str(WINDOW_SIZE))

MODEL_PATH = os.path.join("checkpoints_20220403-220452", "m0403.016-0.19.hdf5")


if sys.platform == "win32":
    
    PATH_DELIM = "\\"
    USE_MULTIP = False
    
elif sys.platform == "linux": 
    
    PATH_DELIM = "/"
    USE_MULTIP = True
    
    
######################################################

    
    




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

######################################################

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
        return int(np.floor(len(self.ids) / self.batch_size)) + 1
    
    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_ids = [self.ids[k] for k in indexes]
        # print("********************************",batch_ids,"********************************")
        # Generate data
        X, y, = self.__data_generation(batch_ids)
        
        
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
            # print(f)
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
    

    
######################################################


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

########################################################################################


def plot_confusion_matrix(y_true, y_pred,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    classes = getClasses()
    np.set_printoptions(precision=2)#digit floats 
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm.shape)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        fig_name = "confusion_matrix_nor7.jpg"
    else:
        print('Confusion matrix, without normalization')
        fig_name = "confusion_matrix7.jpg"

    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(fig_name)
    return ax

######################################################


model = load_model(MODEL_PATH, custom_objects={"f1_m": f1_m,
                                               "precision_m": precision_m,
                                               "recall_m":recall_m})
#print(model.metrics_names)

model.summary()

params = {'batch_size': BATCH_SIZE,
      'window_size': WINDOW_SIZE,
      'n_feature': N_FEATURE,
      'shuffle': False}
    
test_paths = load_ids("test")
# print(test_paths)
n_sample = len(test_paths)
test_generator = denseNet_generator(test_paths, **params)

######################################################
# evaluation
if USE_MULTIP:
    
    score = model.evaluate(test_generator,
                  workers=6, use_multiprocessing=True, verbose=1)
else:
    score = model.evaluate(test_generator, verbose=1)
print(score)


######################################################

#prediction
if USE_MULTIP:
    
    y_pred_one = model.predict(test_generator,
                  workers=6, use_multiprocessing=True)
else:
    y_pred_one = model.predict(test_generator, verbose=1)

y_pred_one = y_pred_one[:n_sample]
print(y_pred_one.shape)

y_pred = y_pred_one.argmax(axis=1)
print(y_pred.shape)
print(y_pred[:50])

######################################################



#get_true_labels
y_true_one = np.array([])

i = 0
for _, batch_y in test_generator:
    #print(i)
    if i == 0:

        y_true_one = batch_y
    else:

        y_true_one = np.concatenate((y_true_one, batch_y))

    i += 1
    
y_true_one = y_true_one[:n_sample]
print(y_true_one.shape)

y_true = y_true_one.argmax(axis=1)
print(y_true.shape)
print(y_true[:50])

######################################################

print(len(test_paths), y_pred.shape, y_true.shape)

for i in range(10):
    
    print(test_paths[i]["path"][-1], y_pred[i], y_true[i])



######################################################
plot_confusion_matrix(y_true, y_pred,
                      normalize=True)
######################################################
plot_confusion_matrix(y_true, y_pred,
                      normalize=False)

