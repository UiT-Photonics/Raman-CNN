# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:16:10 2023

@author: mje059
"""

import os
import numpy as np
import tensorflow as tf
from CVAE import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def mergeclass(merge,oldclass,oldkey):
    replaceclass = min(merge)
    
    join = [i for i in merge if i!=replaceclass]
    
    newclass = np.copy(oldclass)
    newkey = {}
    for n in list(oldkey.keys()):
        newkey[n] = oldkey[n]
        
    for i in join:
        I = (newclass[:,0] == i)
        newclass[I,0] = replaceclass
        newkey.pop(i)
    return [newclass, newkey]

def squeezeclass(oldformat,oldkey):
    newkey = {}
    oldlabels = np.unique(oldformat[:,0])
    newlabels = np.asarray(range(0,len(oldlabels)))
    
    newformat = np.copy(oldformat)
    for i in newlabels:
        o = oldlabels[i]
        newkey[i] = oldkey[o]
        I = (oldformat[:,0]==o)
        newformat[I,0] = i
    return [newformat,newkey]

def popclass(pop,olddata,oldclass,oldkey):
    newkey = {}
    
    classes = list(oldkey.keys())
    
    newclass = np.zeros((1,2))
    newdata = np.zeros((1,olddata.shape[1]))
    
    for i in classes:
        if i not in pop:
            I = (oldclass[:,0]==i)
            newclass = np.concatenate((newclass,oldclass[I,:]),axis=0)
            newdata = np.concatenate((newdata,olddata[I,:]),axis=0)
            newkey[i] = oldkey[i]
            
    return [newdata[1:],newclass[1:],newkey]


class EVClassNN():
    def __init__(self):
        #Parameters of network
        
        #Number of neurons in each layer
        self.Neurons = [512,256,256,128,128,128]
        #Dropout rate for each layer
        self.Dropout = [0.0]*len(self.Neurons)
        #Flag for the use of batch normalization
        self.Use_BatchNorm = True
        #Number of classes in data
        self.Classes = 9
        #Number of latent dimensions to accept
        self.LatentDims = 110
        #Flag for use of explicit x in CVAE or not (affects number of true data in latent space)
        self.ExplicitX = True
        
        self.lr=1e-2
        
        #Class weight dictionary
        self.cw = {}
    
    def buildmodel(self):
        
        #Build class weight dictionary
        for i in range(0,self.Classes):
            #Check if there is an entry for the specific class in the dictionary
            if not i in self.cw:
                #If not, add entry to dictionary and scale it to unity
                self.cw.update({i:1})
        
        #Handler for explicit X in CVAE
        if not self.ExplicitX:
            #If X is explicitly considered by CVAE, X-data is included in latent
            #Otherwise there are 5 tag-along paramaters that should be ignored
            hidden = tf.keras.Input(shape=(self.LatentDims+5))
            #Split the unwanted parameters out and continue with only true latent data
            x, __ = tf.split(hidden, [self.LatentDims,5],axis=1)
        else:
            #Else, accept all latent dimensions as input
            hidden = tf.keras.Input(shape=(self.LatentDims))
            x, __ = tf.split(hidden, [self.LatentDims-10,10],axis=1)
        
        #If BatchNorm is flagged, apply batch normalization before FC
        if self.Use_BatchNorm:
            x = tf.keras.layers.BatchNormalization()(x)
        #Build network
        #Add FC layer as prescribed by self.Neurons
        x=tf.keras.layers.Dense(units=self.Neurons[0],activation='linear',kernel_initializer=tf.keras.initializers.HeNormal)(x)
        #Add activation, advanced activation LeakyReLU used here as well
        x=tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        #Add dropout layer
        x=tf.keras.layers.Dropout(self.Dropout[0])(x)
        
        for i in range(1,len(self.Neurons)):
            if self.Use_BatchNorm:
                x = tf.keras.layers.BatchNormalization()(x)
            #z=x
            #Add FC layer as prescribed by self.Neurons
            x=tf.keras.layers.Dense(units=self.Neurons[i],kernel_initializer=tf.keras.initializers.HeNormal,activation='tanh')(x)
            #Add activation, advanced activation LeakyReLU used here as well
            #x=tf.keras.layers.LeakyReLU(alpha=0.01)(x)
            #Add dropout layer
            x=tf.keras.layers.Dropout(self.Dropout[i])(x)
            #x=tf.keras.layers.add([x,z])
    
        
        #Final layer in classification head
        y = tf.keras.layers.Dense(units=self.Classes) (x)
        #Apply softmax to values and pass as output
        output = tf.keras.layers.Softmax(axis=1)(y)
        
        #Build and compile model with crossentropy loss and adam optimizer
        self.Model = tf.keras.Model(hidden,output,name="Classifyer")
        
        self.Model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(self.lr),
            metrics = 'accuracy'
            )
        
    def train(self,data,labels,epochs=50,BS=50,valdata = False):
        #Basic training over dataset (data) with one-hot labels (labels)
        if valdata is None:
            return self.Model.fit(x=data,y=labels,epochs=epochs,batch_size=BS)
        else:
            return self.Model.fit(x=data,y=labels,validation_data = valdata,epochs=epochs,batch_size=BS)
        
    def __call__(self,Data,NetObj=None):
        if NetObj==None:
            Pred = self.Model.predict(Data)
            
        elif str(type(NetObj))=="<class 'CVAE.Raman_CVAE'>":
            Lat = NetObj.encoder(Data)
            Pred = self.Model.predict(Lat)
            
        return np.argmax(Pred,axis=1)
    
    def adapt_to_Net(self,NetObj):
        if str(type(NetObj))=="<class 'CVAE.Raman_CVAE'>":
            self.ExplicitX=NetObj.use_explicit_shift
            self.LatentDims=NetObj.hiddendim
        self.buildmodel()
    
    def adapt_to_data(self,data,labels,cwunity=True,explicitX=False,softclampweight=10):
        if explicitX:
            self.LatentDims = data.shape[1]
            self.ExplicitX  = True
        else:
            self.LatentDims = data.shape[1]-5
            self.ExplicitX = False
        
        self.Classes = labels.shape[1]
        if cwunity:
            #Tallying number of samples and changing weights to scale sensitivity to unity
            nSamples = np.sum(labels,axis=0)/np.sum(labels)
            weight = np.nanmax(nSamples)/nSamples
            
            #Soft clamping weight to prevent artefacts
            weight = weight*(weight<(softclampweight-1))+(weight>(softclampweight-1))*(2/(1+np.exp(2*((softclampweight-1)-weight)))+(softclampweight-2))
            self.cw={}
            for i in range(0,len(weight)):
                #Add new weights
                self.cw.update({i:weight[i]})
        else:
            for i in range(0,labels.shape[1]):
                self.cw.update({i:1})
                
        self.buildmodel()
        
    def save_model(self,path=''):
        #Handle for saving trained model to path
        if not os.path.isdir(path):
            make=[path]
            for lvls in range(0,5):
                parent=os.path.dirname(make[-1])
                if not os.path.isdir(parent):
                    make.append(parent)
                else:
                    break
            if lvls==4:
                print('Invalid directory!')
                return False
            else:
                for f in reversed(make):
                    os.mkdir(f)
        if not os.path.isdir(path+r"\Classifyer"):
            os.mkdir(path+r"\Classifyer")
        if path=='':
            path=self.path
        try:
            self.Model.save_weights(path+r'\\Classifyer\\main.h5')
            print('Model successfully saved to: {}'.format(path+r'\\Classifyer'))
            return True
        except:
            print('An error occurred and the model was NOT saved, review path and model')
            return False
                    

def MakeOHSet(data):
    #Make make dataset from list of data "data" with one-hot labels
    
    #Set number of classes as number of sets in list
    nCls = len(data)
    
    #Handler for different data shapes
    
    dshape = [1]
    for i in range(1,len(data[0].shape)):
        dshape.append(data[0].shape[i])
    #Make empty dataset
    DSet = np.zeros(dshape)
    #And labelset
    LSet = np.zeros((1,nCls))
    
    for i in range(0,len(data)):
        #Add new data to dataset
        DSet = np.concatenate((DSet,data[i]),axis=0)
        #Make empty one-hot vector
        OH = np.zeros((1,nCls))
        #Set appropriate entry to one to encode class
        OH[0][i] = 1
        #Make vector into matrix of compatible size with data
        OH = np.tile(OH,[len(data[i]),1])
        #Add one hot matrix to labelset
        LSet = np.concatenate((LSet,OH),axis = 0)
        
    
    return DSet[1:],LSet[1:]

def Confuse(DSet,LSet,NN):
    #Function for determining confusion matrix of predictions made by NN
    #NN: object of class EVClassNN
    #DSet: data matrix of shape Nxhidden(+para), output from CVAE encoder
    #Lset: one-hot label matrix of shape NxCLS
    
    #Get predictions from model
    Pred = NN(DSet)
    
    #Determine number of classes from shape of label matrix
    nCls=LSet.shape[1]
    
    #Pre-allocate confusion matrix
    ConMat = np.zeros((nCls,nCls))
    
    #Iterate through true classes
    for i in range(0,nCls):
        #Determine where samples have true class i
        I_Cls = np.where(LSet[:,i]==1)[0]
        for n in I_Cls:
            #Add entry to ConMat determined by the predicted class
            ConMat[i,Pred[n]]+=1
    
    return ConMat
            
        
        
    
        