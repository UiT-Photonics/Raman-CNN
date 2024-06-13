# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:28:48 2022

@author: mje059
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 03:34:38 2021

@author: mje059
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import RamanData

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Raman_CVAE():
    def __init__(self):
        #######################################################################
        #Dictionaries
        
        #Optimizer options
        self.opts=['Adadelta',
         'Adagrad',
         'Adam',
         'Adamax',
         'Ftrl',
         'Nadam',
         'RMSprop',
         'SGD']
        
        #Initializer options
        self.i_opts=['Constant',
         'GlorotNormal',
         'GlorotUniform',
         'HeNormal',
         'HeUniform',
         'Identity',
         'Initializer',
         'LecunNormal',
         'LecunUniform',
         'Ones',
         'Orthogonal',
         'RandomNormal',
         'RandomUniform',
         'TruncatedNormal',
         'VarianceScaling',
         'Zeros']
        
        #Activation options (primitive)
        self.a_opts=['elu',
         'exponential',
         'gelu',
         'hard_sigmoid',
         'linear',
         'relu',
         'selu',
         'sigmoid',
         'softmax',
         'softplus',
         'softsign',
         'swish',
         'tanh']
        
        #Iterators to translate to functional objects
        self.o_dict=[]
        for i in range(0,len(self.opts)):
            self.o_dict.append(type(eval('tf.keras.optimizers.'+self.opts[i]+'()')))
        
        self.i_dict=[]
        for i in range(0,len(self.i_opts)):
            self.i_dict.append(type(eval('tf.keras.initializers.'+self.i_opts[i]+'()')))
        
        self.a_dict=[]
        for i in range(0,len(self.a_opts)):
            self.a_dict.append(eval('tf.keras.activations.'+self.a_opts[i]))
            
        
        
        #######################################################################
        #Learning parameters
        
        #Flag for explicit consideration of the  x (shift) axis
        self.use_explicit_shift=True
        
        #Beta parameters, relative weight of KL-loss to reconstruction loss
        self.beta=5
        
        #balancer of reconstruction loss and fourier loss (0->r_loss only, 1->f_loss only)
        self.alpha=0.3
        
        #Balancer to assign importance to x-axis
        self.gamma=1e1
        
        #Learning rate
        self.lr=1e-4
        
        #Learning agorithm
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        
        #######################################################################
        #Encoder parameters
    
        self.enc_conv_layers=[6,6,4]
    
        self.enc_conv_filters=[32,64,128]
    
        self.enc_conv_size=[5,3,3]
        
        self.enc_conv_skip=[True]*len(self.enc_conv_filters)
        
        self.enc_conv_BN=[True]*len(self.enc_conv_filters)
    
        self.enc_FC=[512]*4
        
        self.enc_div0=1e-12
    
        self.enc_conv_init=tf.keras.initializers.HeNormal()
    
        self.enc_drop=[0.1]*len(self.enc_FC)
        
        #Desired input and output width
        self.osize=2048
        
        #Number of latent dimensions
        self.hiddendim=110
    
        #Reduction/inflation ratio for each pooling-layer
        self.red=2
        
        #######################################################################
        #Decoder parameters
        
        #self.dec_conv_layers=[6,5,4]
        
        #self.dec_conv_filters=[32,64,128]
        
        #self.dec_conv_size=[3,5,5]
        
        self.dec_conv_layers=[2,3]
        
        self.dec_conv_filters=[64,32]
        
        self.dec_conv_size=[3,3]
        
        self.dec_conv_skip=[True]*len(self.dec_conv_filters)
        
        self.dec_conv_BN=[True]*len(self.dec_conv_filters)
    
        self.dec_FC=[512]*4
    
        self.dec_conv_init=tf.keras.initializers.HeNormal()
        
        self.dec_drop=[0.1]*len(self.dec_FC)
        
        #######################################################################
        #General modifiers
        
        #Variance limit
        self.Vmag=10
        
        #Frequency mask
        #I=np.tile(np.expand_dims(np.asarray(range(0,self.osize)),1),[1,2])
        self.epsilon = 0.01
        I=np.asarray(range(0,self.osize))
        self.Fmask=np.zeros(I.shape)+(np.exp(-(I)**2/(10*self.osize)))+(np.exp(-(I-self.osize)**2/(10*self.osize)))+self.epsilon*(np.exp(-(I-self.osize/2)**2/(0.01*self.osize**2)))
        
        #Optional flat frequency mask
        #self.Fmask=np.ones(I.shape)
        self.Fmask=self.Fmask/np.max(self.Fmask,axis=0)
        
        #######################################################################
        #Build and environment parameters
        
        #Print summary of build
        self.verbosebuild=True
        
        #'Home' path of process
        self.path=r'C:\\Ramtemp'
        
        #Build on init
        #self.buildmodel()
    
    def __call__(self,D):
        #Funciton to enable the class to be called, e.g. y=Raman_CVAE(x)
        encoded = self.encoder(D)
        reconstructed = self.decoder(encoded)
        return reconstructed
    
    def buildmodel(self):
        #Determine vector size needed for compatability with number of pooling layers
        self.rs=((self.osize)//self.red**len(self.enc_conv_layers)+1)*self.red**len(self.enc_conv_layers)
        
        #Encoder build
        
        
        #Declare encoder input
        enc_input=tf.keras.Input(shape=(self.osize,2))
        
        #Stretches data vector to fit required vector size
        x=StretchLayer(self.rs,1)(enc_input)
        
        #Splits x (spec_in) and y (shift_in) so that they can be piped through different processes
        shift_in, spec_in = tf.split(x,2,axis=2)
        
        #Removes excess dimension from shift_in (Nxnx1->Nxn)
        shift_in = tf.math.reduce_sum(shift_in,axis=2)
        
        #Fits fourth degree polynomial instead of passing entire data
        sh_para=FitPolyLayer(size=self.rs,degree=4)(shift_in)
        
        #Transfers pointer for looped build
        x=spec_in
        
        #Loop for building the convolution blocks of the encoder
        for i in range(0,len(self.enc_conv_layers)):
            #Procedure for making the skip connecion throught the tensor "z"
            if self.enc_conv_skip[i]:
                #z=StretchLayer(x.shape[1],self.enc_conv_filters[i])(x,ch=True)
                z=tf.keras.layers.Conv1D(filters=self.enc_conv_filters[i],kernel_size=1,strides=1,padding="same")(x)
            if self.enc_conv_BN[i]:
                #Option for applying batch normalization
                x=tf.keras.layers.BatchNormalization()(x)
                
            #Convolutional layer with number of filters, kernel size and initializer given by class parameters
            #Forces padding to same size for compatability with input to block such that skip-connections can be used
            x=tf.keras.layers.Conv1D(filters=self.enc_conv_filters[i],kernel_size=self.enc_conv_size[i],strides=1,padding="same",kernel_initializer=self.enc_conv_init)(x)
            
            #Using separate activation layer so that LeakyReLu can be used
            x=tf.keras.layers.LeakyReLU(alpha=0.01)(x)
            for n in range(1,self.enc_conv_layers[i]):
                #Keep building layers within block
                x=tf.keras.layers.Conv1D(filters=self.enc_conv_filters[i],kernel_size=self.enc_conv_size[i],strides=1,padding="same",kernel_initializer=self.enc_conv_init)(x)
                x=tf.keras.layers.LeakyReLU(alpha=0.01)(x)
                
            if self.enc_conv_skip[i]:
                #Implementing skip re-connect
                x=tf.keras.layers.add([x,z])
            
            #Terminate block with MaxPool layer
            x=tf.keras.layers.MaxPooling1D(pool_size=self.red,padding="valid")(x)
            
        #Flatten output from convolutional section to make it compatible with FC-block(s)
        x=tf.keras.layers.Flatten()(x)
        
        if self.use_explicit_shift:
            #"Explicit X" sub-net
            meta = tf.keras.layers.Dense(units=64,kernel_initializer='HeNormal')(sh_para)
            meta = tf.keras.layers.LeakyReLU(alpha=0.01)(meta)
            meta = tf.keras.layers.Dense(units=64,kernel_initializer='HeNormal')(meta)
            meta = tf.keras.layers.LeakyReLU(alpha=0.01)(meta)
            meta = tf.keras.layers.Dense(units=64,kernel_initializer='HeNormal')(meta)
            meta = tf.keras.layers.LeakyReLU(alpha=0.01)(meta)
            #If shift is to be explicitly considered, this is added to the flattened data such that it can be directly processed by FC block(s)
            x=tf.concat((x,meta),axis=1)
        
        for i in range(0,len(self.enc_FC)):
            #Build FC-layer with parameters declared during __init__
            x=tf.keras.layers.Dense(units=self.enc_FC[i],kernel_initializer='HeNormal')(x)
            x=tf.keras.layers.LeakyReLU(alpha=0.01)(x)
            #Add option for dropout
            x=tf.keras.layers.Dropout(self.enc_drop[i])(x)
        
        #Final encoder layer with linear mean activation and relu with div0 prevention on variance
        #Encoder output given by non-activated Dense for means, initializer set to zero to clamp initial guesses and limit delta values
        enc_mu=tf.keras.layers.Dense(units=self.hiddendim,kernel_initializer='HeNormal')(x)
        
        #Variance output of encoder, enc_div0 is added to prevent variance from becoming zero and Vmag is factored in to prevent extreme variances
        enc_V=self.enc_div0+self.Vmag*tf.keras.layers.Dense(units=self.hiddendim,activation='sigmoid',kernel_initializer='zeros')(x)
        
        #Variational encoder is built
        self.encoder_mu_V=tf.keras.Model(enc_input,(enc_mu,enc_V),name='Encoder_mu_V')
        
        #Sampling function is used to connect enc_mu and enc_V back to main pipeline
        x=tf.keras.layers.Lambda(self.sampling)([enc_mu,enc_V])
        
        
        if self.use_explicit_shift:
            #If x-axis is explicitly considered, its information will be contained within x and is passed to the latent space
            enc_output=x
        else:
            #If x-axis is not considered, fit parameters determined earlier are passed as piggyback onto latent space
            enc_output=tf.concat((x,sh_para),axis=1)
        
        #Encoder is declared and built
        self.encoder=tf.keras.Model(enc_input,enc_output,name="Encoder")
        
        #######################################################################
        #Decoder build
        
        #Input from latent space
        if self.use_explicit_shift:
            #If x-axis is explicitly considered, the entire latent space is passed as input
            dec_input=tf.keras.Input(shape=self.hiddendim)
            y, sh_para_rec = tf.split(dec_input,[self.hiddendim-10,10],axis=1)
            
            meta = tf.keras.layers.Dense(units=64,kernel_initializer='HeNormal')(sh_para_rec)
            meta=tf.keras.layers.LeakyReLU(alpha=0.01)(meta)
            meta = tf.keras.layers.Dense(units=64,kernel_initializer='HeNormal')(meta)
            meta=tf.keras.layers.LeakyReLU(alpha=0.01)(meta)
            meta = tf.keras.layers.Dense(units=64,kernel_initializer='HeNormal')(meta)
            meta=tf.keras.layers.LeakyReLU(alpha=0.01)(meta)
            
            y = tf.concat((y,meta), axis=1)
            
        else:
            #If x-axis is not considered, extract 5 parameters of shift axis poly and pass them as separate from x
            dec_input=tf.keras.Input(shape=self.hiddendim+5)
            y, sh_para_rec = tf.split(dec_input,[self.hiddendim,5],axis=1)

        
        for i in range(0,len(self.dec_FC)):
            #Build FC block
            #Dense layer of decoder with activation
            y=tf.keras.layers.Dense(units=self.dec_FC[i],kernel_initializer='HeNormal')(y)
            y=tf.keras.layers.LeakyReLU(alpha=0.01)(y)
        
            #Add dropout layer for future use
            y=tf.keras.layers.Dropout(self.dec_drop[i])(y)
        
        ##CHECK THAT FS>=self.dec_FC[-1]
        fs=np.max([self.rs//(2**len(self.dec_conv_layers)),self.dec_FC[-1]//8,self.hiddendim])
        
        
        
        if self.use_explicit_shift:
            y=tf.keras.layers.Dense(units=fs+5,kernel_initializer='HeNormal')(y)
            y, sh_para_rec = tf.split(y,[fs,5],axis=1)
        else:
            y=tf.keras.layers.Dense(units=fs,kernel_initializer='HeNormal')(y)
        
        y=tf.keras.layers.Reshape((fs,1))(y)
        
        for i in range(0,len(self.dec_conv_layers)):
            y=tf.keras.layers.UpSampling1D(size=self.red)(y)
            if self.dec_conv_skip[i]:
                z=tf.keras.layers.Conv1D(filters=self.dec_conv_filters[i],kernel_size=2*self.red+1,padding="same")(y)
                #z=StretchLayer(y.shape[1],self.dec_conv_filters[i])(y,ch=True)
                #n = self.dec_conv_filters[i]//y.shape[2]+1
                #z = tf.math.reduce_sum(x,axis=2,keepdims=True)
                #z = tf.tile(y, (1,1,n))
                #z = z[:,:,0:self.dec_conv_filters[i]]
                
            #Build conv layer
            #Upsampling to counter MaxPool in encoder
            #Use UpSampling1D for nearest neighbour
            
            #y=tf.keras.layers.Conv1D(filters=y.shape[-1],kernel_size=2*self.red+1,padding="same")(y)
            
            #Use NN upsample followed by normal conv instead of DeConv to suppress artefacts
                
            if self.dec_conv_BN[i]:
                y=tf.keras.layers.BatchNormalization()(y)
            
            #Deconvolution
            y=tf.keras.layers.Conv1D(filters=self.dec_conv_filters[i],kernel_size=self.dec_conv_size[i],strides=1,padding="same",kernel_initializer=self.dec_conv_init)(y)
            y=tf.keras.layers.LeakyReLU(alpha=0.01)(y)
            
            for n in range(1,self.dec_conv_layers[i]):
                #Build convolution block, same as in encoder
                y=tf.keras.layers.Conv1D(filters=self.dec_conv_filters[i],kernel_size=self.dec_conv_size[i],strides=1,padding="same",kernel_initializer=self.dec_conv_init)(y)
                y=tf.keras.layers.LeakyReLU(alpha=0.01)(y)
                
                
            if self.dec_conv_skip[i]:
                y=tf.keras.layers.add([y,z])
                
                
        #Effectively summation operation over data
        y=tf.math.reduce_sum(y,axis=2,keepdims=True)
       
        if self.use_explicit_shift:
            #Use mediator to reduce sensitivity of higher degrees of the polynomial, avoiding chaotic delta's during training
            self.mediator = tf.constant([1e-11,1e-7,1e-3,1e+1,1e+3])
            sh_para_rec = self.mediator*tf.keras.layers.Dense(units=5,activation='tanh',kernel_initializer='zeros')(sh_para_rec)
        
        #Use determined parameters to recreate shift axis, for non-explicit case use piggybacked parameters passed through latent space
        sh=MakePolyLayer(size=self.rs,degree=4)(sh_para_rec)
        #Expand dimensions such that the x and y axis are compatible
        sh=StretchLayer(self.osize,1)(tf.expand_dims(sh,axis=2))
        y=StretchLayer(self.osize,1)(y)
        
        #Merge shift and intensity back together
        y=tf.concat((sh,y),axis=2)
        
        #Stretch output back to original size for comparison compatability
        dec_output=y
            
        #Declare and build decoder model
        self.decoder=tf.keras.Model(dec_input,dec_output,name="Decoder")
        
        #Declare autoencoder input
        ae_input=tf.keras.Input(shape=(self.osize,2),name="spec")
        
        #Declare comparison input to autoencoder
        comp=tf.keras.Input(shape=(self.osize,2),name="target")
    
        #Describe autoencoder pipeline
        encoded=self.encoder(ae_input)
        
        decoded=self.decoder(encoded)
        
        #Declare autoencoder (mainmodel "CVAE")
        self.mainmodel=tf.keras.Model([ae_input,comp],decoded,name='CVAE')
        
        #Add custom loss function
        self.mainmodel.add_loss(self.loss_func_fft(ae_input,comp,decoded))
        self.mainmodel.compile(
            loss=None,
            optimizer=self.optimizer,
            )
        print('Model build successful, summary:')
        self.mainmodel.summary()
        
        if self.verbosebuild:
            self.encoder.summary()
            self.decoder.summary()
            
    #@tf.function
    def loss_func_fft(self, x, y_true, y_predict):
        
        
        #Intermediate output with variance and mean
        encoder_mu, encoder_variance = self.encoder_mu_V(x)
            
        #Basic loss, mean square error of reconstruction vs. target
        #Gives lower loss the closer the prediction is to the true
        sd = tf.math.reduce_mean(tf.keras.backend.square(y_true-y_predict),axis=1)
        
        reconstruction_loss = tf.keras.backend.sqrt(tf.math.reduce_mean(sd*[1,self.gamma], axis=[1]))
      
        #Distribution loss, KL divergence of the latent variables
        #Gives lower loss the closer the distriution  is to a normal curve
        kl_loss = -0.5 * tf.math.reduce_sum(1.0 + tf.keras.backend.log(encoder_variance) - tf.keras.backend.square(encoder_mu) - encoder_variance,axis=1)
        
        #Split predicted and true into x and y
        y_true_sh, y_true_A = tf.split(y_true,2,axis=2)
        
        y_predict_sh, y_predict_A = tf.split(y_predict,2,axis=2)
        
        #Total spatial loss function
        #L_s = reconstruction_loss + self.beta*kl_loss
        L_s = reconstruction_loss + self.beta*kl_loss
        
        #Fourier transform of true vs. predicted
        Fy_true=tf.signal.fft(tf.cast(y_true_A,'complex128'))*self.Fmask
        
        Fy_predict=tf.signal.fft(tf.cast(y_predict_A,'complex128'))*self.Fmask
        
        #Fourier loss
        L_f = (tf.math.reduce_mean(tf.keras.backend.square(tf.math.imag(Fy_predict-Fy_true)),axis=[1,2])+
               tf.math.reduce_mean(tf.keras.backend.square(tf.math.real(Fy_predict-Fy_true)),axis=[1,2]))
        
        
        return self.alpha*tf.cast(L_s,'float32')+(1-self.alpha)*tf.cast(L_f,'float32')
            
    def sampling(self,mu_log_variance):
        #Sampling using stochastic variable
        
        #Extact mean and logVar
        mu, log_variance = mu_log_variance
        
        #Generate samples from standard normal distribution
        epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
        
        #Alter mean and variance to match mean and logVar given by encoder
        random_sample = mu + tf.keras.backend.exp(log_variance/2) * epsilon
        return random_sample
    
    def train(self,spec,target,epochs=50,BS=20,trate=0.8):
        #Generic training function
        h=self.mainmodel.fit(x=[spec.astype('float32'),
                          target.astype('float32')],
                             y=None,
                          batch_size=int(BS),
                          epochs=int(epochs),
                          validation_split=1-trate)
        return h
    
    def scramTrain(self,DataStruct,
                   savepath='',
                   testrate=0.2,
                   epochs=100,
                   SoftStart=True,
                   PreTrain=0,
                   BS=20,
                   N_sam = 0,
                   lstep=1,
                   LoadBest=True,
                   Xnoise_rate = 1,
                   Ynoise_rate = 1,
                   Clip_rate = 1,
                   Zero_rate = 1,
                   True_rate = 1,
                   Xnoise_mag = 1,
                   Ynoise_mag = -9,
                   Cliplims = [800,1500],
                   ):
        #More specialized training function
        #Data: DataStruct with spectra (RamanData class)
        
        #Training parameters
        #savepath: path for saving model and parameters, defaults to class path
        #testrate: split ratio of test to train samples
        #epochs: number of epochs to be used for the randomized training
        #SoftStart: flag for easing into training using multiple epochs in the early cycles of training
        #Pretrain: number of epochs to be trained on the clean data only
        #BS: batch size
        
        #Data parameters
        #N_sam: number of spectra to be generated for each training epoch
        #lstep: learning steps, used when reducing learning rate is implemented
        #LoadBest: load last best network on crash/nan-values
        #Xnoise_rate: rate of samples with X-noise (calibration error) as compared to other noise sources
        #Ynoise_rate: rate of samples with Y_noise (gaussian noise) as compared to other noise sources
        #Clip_rate: rate of clipped samples (acquisition range) compared to other noise sources
        #Zero_rate: rate of zero-signal samples compared to other noise sources
        #True_rate: rate of samples with X_noise as compared to other noise sources
        #Xnoise_mag: magnitude of calibration error emulation in % of real
        #Ynoise_mag: magnitude of noise in spectra in dB of RMS
        #Cliplims: the highest starting point and lowest end point in cm^-1
        
        #Initializer to prevent excessive console returns
        os.environ['AUTOGRAPH_VERBOSITY'] = '0'
        
        #Isolate the test and train sets
        if 'train' not in DataStruct.metalists:
            DataStruct.isolate_test_train(exclude=['SiOBg'],mode='per')
        
        #Configure noise parameters
        DataStruct.Xnoise_mag = Xnoise_mag
        DataStruct.Ynoise_mag = Ynoise_mag
        DataStruct.Cliplims = Cliplims
        
        if savepath=='':
            savepath=self.path+r'\\interrim'
        loss_test=[]
        loss_train=[]
        
        print('--------------------------------------------------')
        print('Initial data generated')
        Set = DataStruct.make_mod_set('train',
                                      Xnoise_rate = Xnoise_rate,
                                      Ynoise_rate = Ynoise_rate,
                                      Clip_rate = Clip_rate,
                                      Zero_rate = Zero_rate,
                                      True_rate = True_rate,
                                      ngen = 100,
                                      giveclean = True)
        
        TSet = DataStruct.make_mod_set('test',
                                      Xnoise_rate = Xnoise_rate,
                                      Ynoise_rate = Ynoise_rate,
                                      Clip_rate = Clip_rate,
                                      Zero_rate = Zero_rate,
                                      True_rate = True_rate,
                                      ngen = 100,
                                      giveclean = True)
        
        loss_test.append(self.mainmodel.evaluate(x=[TSet[0],TSet[1]],y=None,batch_size=BS,verbose=0))
        loss_train.append(self.mainmodel.evaluate(x=[Set[0],Set[1]],y=None,batch_size=BS,verbose=0))
        print('Initial loss on training set: {}'.format(loss_train[0]))
        print('Initial loss on test set: {}'.format(loss_test[0]))
        
        if PreTrain>0:
            print('Beginning pre-training on clean data')
            trainData = DataStruct.make_mod_set('train',Xnoise_rate=0,Ynoise_rate=0,Clip_rate=0,Zero_rate=0)
            self.train(trainData,trainData,epochs=PreTrain,BS=BS)
            print('Pre-training complete, beginning scramble-training')
            
        
        if SoftStart:
            print('Soft start enabled, initial cycles will run multiple epochs pr. set')
            n=epochs-8
            eps=[int(6.25*0.8*np.exp(-0.8*i)+1) for i in range(0,n)]
        else:
            eps=[1]*epochs
        for l in range(0,lstep):
            for cycle in eps:
                print('--------------------------------------------------')
                print('New scramble cycle beginning, new noisy set generated')
                #Make set for this cycle using the parameters perscribed in the method call
                #invoke "giveclean" in method such that Set[0] is a matrix with noise and Set[1] is a matrix wthout noise
                Set = DataStruct.make_mod_set('train',
                                              Xnoise_rate = Xnoise_rate,
                                              Ynoise_rate = Ynoise_rate,
                                              Clip_rate = Clip_rate,
                                              Zero_rate = Zero_rate,
                                              True_rate = True_rate,
                                              ngen = N_sam,
                                              giveclean = True)
                
                TSet = DataStruct.make_mod_set('test',
                                              Xnoise_rate = Xnoise_rate,
                                              Ynoise_rate = Ynoise_rate,
                                              Clip_rate = Clip_rate,
                                              Zero_rate = Zero_rate,
                                              True_rate = True_rate,
                                              ngen = N_sam,
                                              giveclean = True)
                
                if cycle>1:
                    print('Beginning cycle {} of {}, training over {} epochs on scrambled set'.format(len(loss_test)+1,len(eps),cycle))
                else:
                    print('Beginning cycle {} of {}, training over single epoch on scrambled set'.format(len(loss_test)+1,len(eps)))
                print('--------------------------------------------------')
                cycle_h=self.train(Set[0],Set[1],epochs=cycle,BS=BS)
                if np.sum(np.where(np.isnan(cycle_h.history.get('loss')), 1, 0))>0:
                    print('Loss returned nan!')
                    print('Returning to last good model')
                    self.verbosebuild=False
                    self.load_model(path=savepath)
                
                    continue
                    
                
                
                loss_test.append(self.mainmodel.evaluate(x=[TSet[0],TSet[1]],y=None,batch_size=BS,verbose=0))
                loss_train.append(np.mean(cycle_h.history.get('loss')))
                print('--------------------------------------------------')
                if len(loss_test)==1:
                    print('Cycle complete, initial loss on training set: {}'.format(loss_train[0]))
                    print('Cycle complete, initial loss on test set: {}'.format(loss_test[0]))
                else:
                    print('Cycle complete, new loss on training set: {} ({}% reduction)'.format(loss_train[-1],round(100*(1-loss_train[-1]/loss_train[-2]),1)))
                    print('Cycle complete, new loss on test set: {} ({}% reduction)'.format(loss_test[-1],round(100*(1-loss_test[-1]/loss_test[-2]),1)))
                fig, axs = plt.subplots(2)
                fig.suptitle('Cycle learning outcome')
                axs[0].plot(np.asarray(range(0,len(loss_train))),loss_train,'k')
                axs[0].plot(np.asarray(range(0,len(loss_test))),loss_test,'b')
                axs[0].set(ylabel='Loss',xlabel='Cycle')
                axs[0].set_yscale('log')
                
                I_example=np.random.choice(range(0,len(Set[0])),1)
                y_true = Set[1][I_example]
                y_noise = Set[0][I_example]
                enc = self.encoder(Set[0][I_example])
                y_predict = self.decoder(enc)
                axs[1].plot(y_true[0,:,0],y_true[0,:,1],'b')
                axs[1].plot(y_noise[0,:,0],y_noise[0,:,1],'r')
                axs[1].plot(y_predict[0,:,0],y_predict[0,:,1],'k')
                axs[1].set(ylabel='Intensity',xlabel='Shift (cm-1)')
                plt.show()
                print('Saving model to: {}'.format(savepath))
                if loss_test[-1]==min(loss_test):
                    self.save_model(path=savepath+r'\\best')
                else:
                    self.save_model(path=savepath)
            if LoadBest:
                print('Learning step completed, loading best model')
                self.verbosebuild=False
                self.load_model(path=savepath+r'\\best')
            else:
                print('Learning step completed')
            if lstep>(l+1):
                print('Starting new learning step, reducing learning rate by factor 10')
                self.lr*=0.1
                print('New learning rate: {}'.format(self.lr))
                
        return [loss_test,loss_train]
        
            
    def load_model(self,path=''):
        #Handle for loading trained model from path
        if path=='':
            path=self.path
        if os.path.isdir(path):
            l=os.listdir(path)
            if not 'main.h5' in l:
                print('Weights of main model missing!')
                return False
            if not 'latgen.h5' in l:
                print('Missing latent space distribution weights!')
                return False
            if not 'para.CVAEc' in l:
                print('Missing configuration data!')
                return False
            
            self.Load_config(path=path)
            self.mainmodel.load_weights(path+r'\\main.h5')
            self.encoder_mu_V.load_weights(path+r'\\latgen.h5')
        else:
            return False
            
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
                
        if path=='':
            path=self.path
        try:
            self.mainmodel.save_weights(path+r'\\main.h5')
            self.encoder_mu_V.save_weights(path+r'\\latgen.h5')
            self.Save_config(path=path)
            print('Model successfully saved to: {}'.format(path))
            return True
        except:
            print('An error occurred and the model was NOT saved, review path and model')
            return False
        
    def get_config(self):
        parsed=[]
        
        parsed.append(self.hiddendim)
        parsed.append(self.red)
        
        I_o=[i for i in range(0,len(self.o_dict)) if type(self.optimizer)==self.o_dict[i]][0]
        parsed.append(self.opts[I_o])
        
        parsed.append(self.lr)

        parsed.append(self.enc_conv_layers)

        parsed.append(self.enc_conv_filters)
        
        parsed.append(self.enc_conv_size)
        
        parsed.append(self.enc_conv_skip)
        
        parsed.append(self.enc_FC)
        
        I_i=[i for i in range(0,len(self.i_dict)) if type(self.enc_conv_init)==self.i_dict[i]][0]

        parsed.append(self.i_opts[I_i])
        

        parsed.append(self.enc_drop)
    
        parsed.append(self.dec_conv_layers)
        
        parsed.append(self.dec_conv_filters)
        
        parsed.append(self.dec_conv_size)
        
        parsed.append(self.dec_FC)
        
        I_i=[i for i in range(0,len(self.i_dict)) if type(self.dec_conv_init)==self.i_dict[i]][0]

        parsed.append(self.i_opts[I_i])
        
        parsed.append(self.dec_drop)
        return parsed
        
    def set_config(self,conf):
        self.hiddendim=conf.pop(0)
        self.red=conf.pop(0)
        self.optimizer=eval('tf.keras.optimizers.'+conf.pop(0)+'()')
        self.lr==conf.pop(0)
        
        self.enc_conv_layers=list(conf.pop(0))
        self.enc_conv_filters=list(conf.pop(0))
        self.enc_conv_size=list(conf.pop(0))
        self.enc_conv_skip=list(conf.pop(0))
        self.enc_FC=list(conf.pop(0))
        self.enc_conv_init=eval('tf.keras.initializers.'+conf.pop(0)+'()')
        self.enc_drop=list(conf.pop(0))
        
        self.dec_conv_layers=list(conf.pop(0))
        self.dec_conv_filters=list(conf.pop(0))
        self.dec_conv_size=list(conf.pop(0))
        self.dec_FC=list(conf.pop(0))
        self.dec_conv_init=eval('tf.keras.initializers.'+conf.pop(0)+'()')
        self.dec_drop=list(conf.pop(0))
        
        self.buildmodel()
    
    def Save_config(self,path=''):
        if path=='':
            path=self.path
        config=self.get_config()
        with open(path+r'\\para.CVAEc','w',newline='') as file:
            for en in config:
                file.write(str(en)+'\n')
                
    def Load_config(self,path=''):
        if path=='':
            path=self.path
        with open(path+r'\\para.CVAEc','r') as file:
            config_r=[line.rstrip('\n') for line in file]
            
        config=[]
        for entry in config_r:
            try:
                config.append(int(entry))
            except:
                try:
                    config.append(float(entry))
                except:
                        
                    if '[' in entry:
                        tmp=[]
                        tmp=entry.split(',')

                        tmp[0]=tmp[0][1:]
                        tmp[-1]=tmp[-1][:-1]
                        
                        try:
                            for i in range(0,len(tmp)):
                                tmp[i]=int(tmp[i])
                            config.append(tmp)
                        except:
                            config.append(['True' in i for i in tmp])
                        
                    else:
                        config.append(entry)
                        
        self.set_config(config)
            
class StretchLayer(tf.keras.layers.Layer):
  def __init__(self, x_length,y_length,dtype='float32'):
    super(StretchLayer, self).__init__()
    self.x_length = x_length
    self.y_length = y_length
    self.format = dtype

  def call(self, x,ch=False):
    #Add a dimension to make it an "image" such that it is compatible with the Resize layer
    if ch:
        #Real shady stuff going on in here
        dim_pad=tf.expand_dims(x,axis=3)
        stretch = tf.keras.layers.Resizing(self.x_length,self.y_length,interpolation='nearest')(dim_pad)
        
        recut = tf.math.reduce_sum(stretch,axis=3)
        
        return tf.cast(recut,self.format)
    if len(x.shape)==3:
        dim_pad=tf.expand_dims(x,axis=2)
    elif len(x.shape)==2:
        dim_pad=tf.expand_dims(x,axis=0)
        dim_pad=tf.expand_dims(dim_pad,axis=2)
    elif len(x.shape)==1:
        dim_pad = tf.expand_dims(x,axis=1)
        z = tf.zeros_like(dim_pad)
        dim_pad = tf.concat([dim_pad,z],axis=1)
        dim_pad=tf.expand_dims(dim_pad,axis=0)
        dim_pad=tf.expand_dims(dim_pad,axis=2)
    
    #Stretch data through Resizing layer
    stretch=tf.keras.layers.Resizing(self.x_length,1)(dim_pad)
    #Clip away "appendix" dimension to return to standard format
    if len(x.shape)==3:
        recut=tf.math.reduce_sum(stretch,axis=2)
    elif len(x.shape)==2:
        recut=tf.math.reduce_sum(stretch,axis=[0,2])
    elif len(x.shape)==1:
        recut=tf.math.reduce_sum(stretch,axis=[0,2,3])
    
    return tf.cast(recut,self.format)

          
class FitPolyLayer(tf.keras.layers.Layer):
  def __init__(self, size=2048, degree=4,dtype='float32'):
    super(FitPolyLayer, self).__init__()
    self.format = dtype
    
    I=tf.cast(tf.expand_dims(range(0,size),axis=1),dtype='float32')
    
    self.X=tf.ones(I.shape)
    
    #Row n corresponding to nth degree
    for i in range(1,degree+1):
        self.X=tf.concat((I**i,self.X),axis=1)

  def call(self, x):
    
    
    xtx=tf.tensordot(tf.transpose(self.X),self.X,axes=[1,0])
    inv=tf.linalg.inv(xtx)
    fac=tf.tensordot(inv,tf.transpose(self.X),axes=[1,0])
    res=tf.tensordot(x,tf.transpose(fac),axes=[1,0])
    
    return res

class MakePolyLayer(tf.keras.layers.Layer):
  def __init__(self, size=2048, degree=4,dtype='float32'):
    super(MakePolyLayer, self).__init__()
    self.format = dtype
    
    I=tf.cast(tf.expand_dims(range(0,size),axis=1),dtype='float32')
    
    self.X=tf.ones(I.shape)
    
    #Row n corresponding to nth degree
    for i in range(1,degree+1):
        self.X=tf.concat((I**i,self.X),axis=1)

  def call(self, beta):
    y=tf.matmul(beta,tf.transpose(self.X))
    
    return y

class SoftClipLayer(tf.keras.layers.Layer):
  def __init__(self, ub, lb,c=1e15):
    super(MakePolyLayer, self).__init__()
    
    self.ub=ub
    self.lb=lb
    self.c=c

  def call(self, x):
    
    f = x - (1/self.c)*tf.math.log(1+tf.math.exp(self.c*(x-self.ub)))+(1/self.c)*tf.math.log(1+tf.math.exp(-self.c*(x-self.lb)))
    
    return f