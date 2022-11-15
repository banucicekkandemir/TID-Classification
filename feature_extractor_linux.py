# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:23:09 2022

@author: bckandemir2018
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:30:08 2022

@author: bckandemir2018
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:50:36 2022

@author: bckandemir2018
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 17:17:57 2022

@author: bckandemir2018
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 01:20:30 2022

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
from keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import sys
from PIL import Image, ImageOps
import csv
###################################################################################################    

cwd = os.getcwd()

DATASETS_PATH = os.path.join(cwd, "datasets7")
DATA_PATH = os.path.join(DATASETS_PATH, "med_data_v7")
# DATASETS_PATH = os.path.join(cwd, "..", "..", "datasets")
# DATA_PATH = os.path.join(DATASETS_PATH, "med_data_v")
EXT_DIR = "med_features7"
EXT_PATH = os.path.join(DATASETS_PATH, EXT_DIR)

COLOR_SPACE = "RGB"

# MODEL_PATH = "m1229.001-1.17.hdf5"
#MODEL_PATH = r'C:\Users\bckandemir2018\Desktop\modell.hdf5'
MODEL_PATH = os.path.join(cwd, "m0402.052-0.11.hdf5")

if sys.platform == "win32":
    
    PATH_DELIM = "\\"
    USE_MULTIP = False
    
elif sys.platform == "linux": 
    
    PATH_DELIM = "/"
    USE_MULTIP = True
    
    
    
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
###################################################################################################

class denseNet_generator(Sequence):
    
    def __init__(self, paths, batch_size, img_size,
                  n_channel):
        
        self.paths = paths
        self.img_size = img_size
        self.n_channel = n_channel
        self.batch_size = batch_size
        self.classes = getClasses()
        self.shuffle = False
        self.on_epoch_end()
        
        
    def __len__(self):
        #'Denotes the number of batches per epoch'
        #son batch önemli, trainde çok mühim değil zaten shuffle var
        return int(np.floor(len(self.paths) / self.batch_size)) + 1
    
    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_paths = [self.paths[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(batch_paths)

        return X, y
    
    
    def on_epoch_end(self):#'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    
    def __data_generation(self, batch_paths):
        #'Generates data containing batch_size samples'
        
        #sample_shape = (224,224,4)
        # X : (n_samples, 4) rgb + ycbcr_MASK
        # Initialization
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
        label_str=path.split(PATH_DELIM)[-2]
        # label_str = path.split(PATH_DELIM)[-1].split("-")[-1].split(".")[0]
        label = self.classes.index(label_str)


        one_hot_y = to_categorical(label,\
                                    num_classes=len(self.classes))
        
        #print("y.sh:", one_hot_y.shape)
        return one_hot_y
    

###################################################################################################    
    
def predictAndSave(model, data_gen, set_name, video_name, video_length):
    
    
    sample_path = os.path.join(EXT_PATH, set_name, video_name)
    if not os.path.isdir(sample_path):
        os.mkdir(sample_path)
    #prediction
   
    pred = model.predict(data_gen)

    pred = pred[:video_length]
    
    label = np.array([])

    i = 0
    for _, batch_y in data_gen:
        
        if i == 0:

            label = batch_y
        else:

            label = np.concatenate((label, batch_y))

        i += 1

    label = label[:video_length]
    label = label.argmax(axis=1)
    #sample = np.array([pred, label])
    
    #print(sample.shape)
    #print(sample_path)
    
    #[0]:X , [1]:Y
    classes = getClasses()
    for i in range(pred.shape[0]):
        
        fname = video_name + "-{:04d}-".format(i+1) + classes[label[i]] + ".npy"
        fpath = os.path.join(sample_path, fname)
        np.save(fpath, pred[i])
    
    return

def load_paths(): #train / val
    
    if sys.platform == 'win32':
        
        fname = os.path.join(DATA_PATH, "video_frame_lists7")
        
    else:
        
        fname = os.path.join(DATA_PATH, "video_frame_lists_linux7")
        
    
    paths = {"train": {},
             "validation": {},
             "test": {}}
    paths["train"]["operasyon"]=[]
    paths["train"]["orgut"]=[]
    paths["train"]["turkiye"]=[]
    paths["train"]["calismak"]=[]
    paths["train"]["gozalti"]=[]
    paths["train"]["baslamak"]=[]
    paths["train"]["aciklamak"]=[]
    paths["train"]["bakan"]=[]
    paths["train"]["arac"]=[]
    paths["train"]["engel"]=[]
    paths["train"]["haber"]=[]
    paths["train"]["almak"]=[]
    paths["train"]["yakalamak"]=[]
    paths["train"]["soylemek"]=[]
    paths["train"]["hazirlamak"]=[]
    
    paths["test"]["operasyon"]=[]
    paths["test"]["orgut"]=[]
    paths["test"]["turkiye"]=[]
    paths["test"]["calismak"]=[]
    paths["test"]["gozalti"]=[]
    paths["test"]["baslamak"]=[]
    paths["test"]["aciklamak"]=[]
    paths["test"]["bakan"]=[]
    paths["test"]["arac"]=[]
    paths["test"]["engel"]=[]
    paths["test"]["haber"]=[]
    paths["test"]["almak"]=[]
    paths["test"]["yakalamak"]=[]
    paths["test"]["soylemek"]=[]
    paths["test"]["hazirlamak"]=[] 
    
    paths["validation"]["operasyon"]=[]
    paths["validation"]["orgut"]=[]
    paths["validation"]["turkiye"]=[]
    paths["validation"]["calismak"]=[]
    paths["validation"]["gozalti"]=[]
    paths["validation"]["baslamak"]=[]
    paths["validation"]["aciklamak"]=[]
    paths["validation"]["bakan"]=[]
    paths["validation"]["arac"]=[]
    paths["validation"]["engel"]=[]
    paths["validation"]["haber"]=[]
    paths["validation"]["almak"]=[]
    paths["validation"]["yakalamak"]=[]
    paths["validation"]["soylemek"]=[]
    paths["validation"]["hazirlamak"]=[]
    _, _, filenames = next(os.walk(fname))
    for f in filenames:
        
        
        with open(os.path.join(fname,f)) as fin:
            frames = [os.path.join(DATASETS_PATH,row.strip()) for row in list(fin)]
        
        key = frames[0].split(PATH_DELIM)[-3]
        for i in frames:
            if i.split(PATH_DELIM)[-2]=="operasyon":
                paths[key]["operasyon"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)

            if i.split(PATH_DELIM)[-2]=="orgut":
                paths[key]["orgut"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)

            if i.split(PATH_DELIM)[-2]=="turkiye":
                paths[key]["turkiye"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)
            if i.split(PATH_DELIM)[-2]=="calismak":
                paths[key]["calismak"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)
            if i.split(PATH_DELIM)[-2]=="gozalti":
                paths[key]["gozalti"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)
            if i.split(PATH_DELIM)[-2]=="baslamak":
                paths[key]["baslamak"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)
            if i.split(PATH_DELIM)[-2]=="aciklamak":
                paths[key]["aciklamak"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)
            if i.split(PATH_DELIM)[-2]=="bakan":
                paths[key]["bakan"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)
            if i.split(PATH_DELIM)[-2]=="arac":
                paths[key]["arac"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)
            if i.split(PATH_DELIM)[-2]=="engel":
                paths[key]["engel"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)
            # paths[key][video] = frames
            if i.split(PATH_DELIM)[-2]=="haber":
                paths[key]["haber"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)
            if i.split(PATH_DELIM)[-2]=="almak":
                paths[key]["almak"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)
            if i.split(PATH_DELIM)[-2]=="yakalamak":
                paths[key]["yakalamak"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)
            if i.split(PATH_DELIM)[-2]=="soylemek":
                paths[key]["soylemek"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)
            if i.split(PATH_DELIM)[-2]=="hazirlamak":
                paths[key]["hazirlamak"].append(i)
                # print(i.split(PATH_DELIM)[-2],"*************",i)
            # paths[key][video] = frames

    
    #print("last pth:", paths[-1])
    return paths
###################################################################################################


model =tf.keras.models. load_model(MODEL_PATH, custom_objects={"f1_m": f1_m,
                                               "precision_m": precision_m,
                                               "recall_m":recall_m}, compile=False)

paths = load_paths()


###################################################################################################

params = {'batch_size': 128,
      'img_size': (224,224),
      'n_channel': 3}

if not os.path.isdir(EXT_PATH):
    os.mkdir(EXT_PATH)

max_len = 0    
for s in paths.keys():
    
    s_list_win = []
    s_list_linux = []
    
    s_path = os.path.join(EXT_PATH, s)
    if not os.path.isdir(s_path):
        os.mkdir(s_path)
    

    for v in paths[s].keys():
        
        if len(paths[s][v]) > max_len:
            max_len = len(paths[s][v])
            
        print(s, v,len(paths[s][v]))
        
        generator = denseNet_generator(paths[s][v], **params)
        print("ilk döngü generatöre girdi ")
        predictAndSave(model, generator, s, v, len(paths[s][v]))
        
        s_list_win.append([EXT_DIR + "\\" + s + "\\" + v + ".npy"])
        s_list_linux.append([EXT_DIR + "/" + s + "/" + v + ".npy"])
        
        
    with open(os.path.join(EXT_PATH, s + '_paths7.txt'), 'w', newline='') as fout:

        writer = csv.writer(fout)
        writer.writerows(s_list_win)      
        
    with open(os.path.join(EXT_PATH, s + '_paths_linux7.txt'), 'w', newline='') as fout:

        writer = csv.writer(fout)
        writer.writerows(s_list_linux)   
        
        
        
        
        
print("max_len:",max_len)

# def format(value):
#     return "%.7f" % value

# formatted = [[format(v) for v in r] for r in m]
# file.write(str(formatted))

# textfile = open("classWeights_fe_w10.txt", "w")
# for element in list1:
#     textfile.write(float(element) + "\n")
# textfile.close()

# for layer in model.layers:
#     weights = layer.get_weights()

# list1=weights[0]
# textfile = open("classWeights_fe_w10.txt", "w")
# formatted = [[format(v) for v in r] for r in list1]
# textfile.write(str(formatted))
# textfile.close()


# def format(value):
#     return "%.7f" % value
# list1=weights[0]
# textfile = open("classWeights_fe_w10.txt", "w")
# for i in list1:
#     for j in i :
#         textfile.write(format(j)+"\n")
# textfile.close()