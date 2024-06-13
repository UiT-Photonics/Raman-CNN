# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:19:22 2022

@author: mje059
"""
import numpy as np

def balancer(old_data,old_cls):
    n_pcr = check(old_cls)
    n_sam = int(min(n_pcr))
    
    u = np.unique(old_cls[:,0])
    n_cls = len(np.unique(old_cls[:,0]))

    ind_master = np.asarray(range(0,len(old_data)))
    
    new_data = np.zeros([1,old_data.shape[1]])
    new_cls = np.zeros([1,old_cls.shape[1]])
    for i in u:
        I = old_cls[:,0] == i
        
        ind_cls = ind_master[I]
        
        ind_new = np.random.choice(ind_cls,n_sam,replace = False)
        
        new_data = np.concatenate((new_data,old_data[ind_new]),axis = 0)
        new_cls = np.concatenate((new_cls,old_cls[ind_new]),axis = 0)
    return [new_data,new_cls.astype(int)]

def mergeclass(merge, old_cls, old_key):
    new_com = min(merge)
    
    new_cls = np.copy(old_cls)
    new_key = {}
    
    for k in list(old_key.keys()):
        new_key[k] = old_key[k]
    
    merge.pop(merge.index(new_com))
    for i in merge:
        I = old_cls[:,0]==i
        new_cls[I,0] = new_com
        new_key.pop(i)
    
    return [new_cls,new_key]

def squeezeclass(old_cls,old_key):
    
    o = np.unique(old_cls[:,0])
    new_cls = np.copy(old_cls)
    new_key = {}
    for i in range(0,len(o)):
        I = old_cls[:,0]==o[i]
        new_cls[I,0] = i
        new_key[i] = old_key[o[i]]
    return [new_cls,new_key]

def check(cur_cls):
    i = range(0,max(cur_cls[:,0]))
    
    n = np.zeros(len(i))
    
    for l in i:
        n[l] = sum(cur_cls[:,0]==l)
    return n

def nn_interpol_tosize(spect,size = 0):
    
    #Determine largest spectrum in set
    #Needed for allocation of pad array
    nmax = 0
    
    for i in range(0,len(spect)):
        nmax = max([nmax,len(spect[i][:,0])])
    
    #If no specific size is declared, interpolate to largest in set
    if size == 0:
        size = nmax
        
            
    #Padded array to hold the data
    zpm = np.zeros((len(spect),nmax,2))

    n = np.zeros(len(spect))
    #Iterate through list to add to matrix
    for i in range(0,len(spect)):
        d = spect[i]
        #Add spectrum to the beginning of the matrix
        zpm[i,0:len(d),:] = d
        #Repeat last entry for empty spaces left over
        zpm[i,len(d):] = d[-1]
        
        #Keep the true length of each spectrum
        n[i] = len(d)
    
    #Matrix form of the true spectra length
    n_m = np.tile(np.expand_dims(n,1),[1,size])

    #Dummy matrix increasing from 0 to 1 along spectral axis
    one = np.linspace(0,1,size)
    one_m = np.tile(np.expand_dims(one,0),[len(spect),1])

    #Form old index matrix, ascending from 0 to n over nmax indices
    I_old = (one_m*(n_m-1)).astype(np.int32)

    #Form new index matrix, ascending from 0 to nmax over nmax indices
    I_new = np.tile(np.expand_dims(range(0,size),0),[len(spect),1])

    #Another dummy matrix for sample number, ranging from 0 to number of samples
    I = np.tile(np.expand_dims(np.asarray(range(0,len(spect))),1),[1,size])

    #New x and y for spectra
    sh_new = np.zeros([len(spect),size])
    spec_new = np.zeros([len(spect),size])

    #Translate from old to new size
    sh_new[I,I_new] = zpm[I,I_old,0]
    spec_new[I,I_new] = zpm[I,I_old,1]

    sh = np.expand_dims(sh_new,2)
    spec = np.expand_dims(spec_new,2)

    #Concatenate stretched output
    new = np.concatenate((sh,spec),axis=2)
        
    return new


def interpol(x_old,y_old,x_new,mode='nn'):
    #Depreciated
    r = np.tile(np.expand_dims(x_new,axis=0),[len(x_old),1])-np.tile(np.expand_dims(x_old,axis=1),[1,len(x_new)])

    if mode=='nn':
        i_map = np.argmin(np.abs(r),axis=0)
        y_new = y_old[i_map]
        
    elif mode=='lin':
        i_low = np.argmax(r*(r<0)-1e3*(r>0),axis=0)
        i_high = np.argmin(r*(r>0)+1e3*(r<0),axis=0)
        
        x_low = x_old[i_low]
        x_high = x_old[i_high]
        
        y_low = y_old[i_low]
        y_high = y_old[i_high]
        
        div = (x_high-x_low)
        
        y_new = y_low+(y_high-y_low)*(x_new-x_low)/np.where(div==0,1,div)
    else:
        return False
    return y_new

def Bestdims(Lat,LabOH,verboseoutput=False):
    #Function for determining best dimensions for scatter plot of data
    
    G=[0]*LabOH.shape[1]
    Lab=np.argmax(LabOH,axis=1)
    for i in range(0,np.max(Lab)+1):
        I_n=np.where(Lab==i)
        G[i]=Lat[I_n]
    C=[]
    ldims=Lat.shape[1];
    for i in range(0,ldims):
        for n in range(i+1,ldims):
            C.append([i,n])
        
    sc=np.zeros(len(C))
    for i in range(0,len(C)):
        sc_tmp=[]
        for n in range(0,len(G)):
            #All other groups
            exc = [k for k in range(0,len(G)) if k != n]
            G_tmp = [[0,0]]*len(exc)
            for m in range(0,len(exc)):
                if len(G[exc[m]])==0 or len(G[n])==0:
                    sc_tmp.append(1)
                else:
                    G_tmp[m]=[np.nanmean(G[exc[m]][:,C[i][0]]),np.nanmean(G[exc[m]][:,C[i][1]])]
                    sc_tmp.append(MahDist(G[n],G_tmp[m],[C[i][0],C[i][1]]))
        sc[i]=np.prod(sc_tmp)
        
    if verboseoutput:
        return [C[np.nanargmax(sc)],sc,C]
    else:
        return C[np.nanargmax(sc)]
    
def Bestdims3(Lat,LabOH,verboseoutput=False):
    #Function for determining best dimensions for scatter plot of data
    
    G=[0]*LabOH.shape[1]
    Lab=np.argmax(LabOH,axis=1)
    for i in range(0,np.max(Lab)+1):
        I_n=np.where(Lab==i)
        G[i]=Lat[I_n]
    C=[]
    ldims=Lat.shape[1];
    for i in range(0,ldims):
        for n in range(i+1,ldims):
            for m in range(n+1,ldims):
                C.append([i,n,m])
        
    sc=np.zeros(len(C))
    for i in range(0,len(C)):
        sc_tmp=[]
        for n in range(0,len(G)):
            #All other groups
            exc = [k for k in range(0,len(G)) if k != n]
            G_tmp = [[0,0,0]]*len(exc)
            for m in range(0,len(exc)):
                if len(G[exc[m]])==0 or len(G[n])==0:
                    sc_tmp.append(1)
                else:
                    G_tmp[m]=[np.mean(G[exc[m]][:,C[i][0]]),np.mean(G[exc[m]][:,C[i][1]]),np.mean(G[exc[m]][:,C[i][2]])]
                    sc_tmp.append(MahDist3(G[n],G_tmp[m],[C[i][0],C[i][1],C[i][2]]))
        sc[i]=np.prod(sc_tmp)
        
    if verboseoutput:
        return [C[np.nanargmax(sc)],sc,C]
    else:
        return C[np.nanargmax(sc)]
    
def MahDist3(D,P,dims):
    #Returns Mahalanobis distance from distribution D to point P using dimensions dims
    #Accepts data D of format Nxh, where h is number of hidden dims
    #Accepts point P of format xh, where h is number of hidden dims
    
    #Reprocesses data matrix to form [2,N] using dims
    Gs=np.asarray([D[:,dims[0]],D[:,dims[1]],D[:,dims[2]]])
    
    #Mean of distribution of D
    mu = np.asarray([np.mean(Gs[0]),np.mean(Gs[1]),np.mean(Gs[2])])
    
    #Covariance matrix
    cm = np.cov(Gs)
    
    dev = P-mu
    
    S = np.linalg.inv(cm)
    
    return np.sqrt(np.dot(dev.T,np.dot(S,dev)))

def MahDist(D,P,dims):
    #Returns Mahalanobis distance from distribution D to point P using dimensions dims
    #Accepts data D of format Nxh, where h is number of hidden dims
    #Accepts point P of format xh, where h is number of hidden dims
    
    #Reprocesses data matrix to form [2,N] using dims
    Gs=np.asarray([D[:,dims[0]],D[:,dims[1]]])
    
    #Mean of distribution of D
    mu = np.asarray([np.mean(Gs[0]),np.mean(Gs[1])])
    
    #Covariance matrix
    cm = np.cov(Gs)
    
    dev = P-mu
    
    S = np.linalg.inv(cm)
    
    return np.sqrt(np.dot(dev.T,np.dot(S,dev)))

def FitPoly(D,degree=4,check=False):
    #Helper function to fit n-degree polynomial to input
    
    #D=D.numpy()
    #D=D.T
    #Takes pure index as the input (X)
    I=np.expand_dims(range(0,len(D)),axis=1)
    
    #First row corresponding to constant
    X=np.ones(I.shape)
    
    #Row n corresponding to nth degree
    for i in range(1,degree+1):
        X=np.concatenate((I**i,X),axis=1)
    
    #Determine fit parameters beta via matrix operation
    beta=D.T.dot((np.linalg.inv(X.T.dot(X))).dot(X.T).T)
    
    #Extra feature for checking and returning MSE of fit
    if check:
        Y=X.dot(beta)
        MSE = np.mean((D-Y)**2)
        return beta.T, check
    else:
        return beta.T
    
def MakePoly(rng,beta):
    #Dummy index vector for constructing X
    I=np.expand_dims(range(0,rng),axis=1)
    
    #Constructing X as matrix where row n corresponds to nth degree
    X=np.ones(I.shape)
    
    for i in range(1,len(beta)):
        X=np.concatenate((I**i,X),axis=1)
        
    #Produce output as matrix operation
    Y=X.dot(beta)
    
    return Y.T

def CompVec3(D,dim1,dim2,method='prod'):
    D1=np.asarray([D[:,dim1[0]],D[:,dim1[1]],D[:,dim1[2]]])
    D2=np.asarray([D[:,dim2[0]],D[:,dim2[1]],D[:,dim1[2]]])
    
    return D1*D2

def interptosize(data,width):
    #Dummy vector for current axis scale
    Io=np.linspace(0,1,len(data))
    #Dummy vector for new axis scale
    In=np.linspace(0,1,width)
    
    #Check for dual channel input
    if data.shape[-1]==2:
        
        #Split channels
        sh = data.T[0]
        spec = data.T[1]
    
        #Interpolate new channels
        new_sh=np.interp(In,Io,sh).reshape((width,1))
        new_spec=np.interp(In,Io,spec).reshape((width,1))
    
        #Recombine and output
        return np.concatenate((new_sh,new_spec),axis=1)
    else:
        return np.interp(In,Io,data)

def modsize(Dset,width):
    nset=np.zeros((Dset.shape[0],width,Dset.shape[2]))
    for i in range(0,len(Dset)):
        nset[i]=interptosize(Dset[i],width)
    return nset

def movmean(x,span=5):
    mask=np.ones([span])/span
    
    x=x.reshape(len(x))
    
    y=np.convolve(x,mask,mode="same")
    return y

def Normalize(data):
    #Normalizes input to range 0-1
    ma=np.max(data)
    mi=np.min(data)
    
    data=(data-mi)/(ma-mi)
    return data


def GenAndSave(CVAE_obj,EVClass_obj,RamanData_obj,path=''):
    if path =='':
        path=CVAE_obj.path
        
    DataDump = os.path.join(path,'Datadump')
    if not os.path.isdir(DataDump):
        os.mkdir(DataDump)
    
    #Get test and train data
    Dtrain, Ltrain = RamanData_obj.get_from_list(RamanData_obj.TrainSet,Label=True)
    Dtest, Ltest = RamanData_obj.get_from_list(RamanData_obj.TestSet,Label=True)
    
    Ltest_exp=np.copy(Ltest)
    Ltest_exp[:,3]=1*(np.sum(np.asarray([Ltest[:,3],Ltest[:,4],Ltest_exp[:,12]]),axis=0)>0)
    Ltest_exp[:,4:11]=Ltest[:,5:12]
    Ltest_exp=Ltest_exp[:,0:11]
    
    Ltrain_exp=np.copy(Ltrain)
    Ltrain_exp[:,3]=1*(np.sum(np.asarray([Ltrain[:,3],Ltrain[:,4],Ltrain_exp[:,12]]),axis=0)>0)
    Ltrain_exp[:,4:11]=Ltrain[:,5:12]
    Ltrain_exp=Ltrain_exp[:,0:11]
    
    LatTrain = CVAE_obj.encoder(Dtrain)
    LatTest = CVAE_obj.encoder(Dtest)
    
    RecTrain = CVAE_obj.decoder(LatTrain)
    RecTest = CVAE_obj.decoder(LatTest)
    
    RecTrain = RecTrain.numpy()
    RecTest = RecTest.numpy()
    
    RecTrain[:,:,0] = Dtrain[:,:,0]
    RecTest[:,:,0] = Dtest[:,:,0]
    
    CTrain_P = EVClass_obj(LatTrain)
    CTest_P = EVClass_obj(LatTest)
    CTrain_T = np.argmax(Ltrain_exp,axis=1)
    CTest_T = np.argmax(Ltest_exp,axis=1)
    
    ConMat_test = Confuse(LatTest,Ltest_exp,EVClass_obj)
    ConMat_train = Confuse(LatTrain,Ltrain_exp,EVClass_obj)
    
    with open(DataDump+r'\TrainData.csv','w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        k=Dtrain
        for i in k:
            writer.writerow(i[:,0])
            writer.writerow(i[:,1])
            
    with open(DataDump+r'\TestData.csv','w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        k=Dtest
        for i in k:
            writer.writerow(i[:,0])
            writer.writerow(i[:,1])
            
    with open(DataDump+r'\TrainLabels.csv','w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        k=Ltrain_exp
        for i in k:
            writer.writerow(i)

    with open(DataDump+r'\TestLabels.csv','w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        k=Ltest_exp
        for i in k:
            writer.writerow(i)
            
    
    with open(DataDump+r'\TrainRec.csv','w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        k=RecTrain
        for i in k:
            writer.writerow(i[:,0])
            writer.writerow(i[:,1])
            
    with open(DataDump+r'\TestRec.csv','w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        k=RecTest
        for i in k:
            writer.writerow(i[:,0])
            writer.writerow(i[:,1])
            
    with open(DataDump+r'\TestLat.csv','w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        k=LatTest.numpy()
        for i in k:
            writer.writerow(i)
    
    with open(DataDump+r'\TrainLat.csv','w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        k=LatTrain
        for i in k:
            writer.writerow(i)
            
    with open(DataDump+r'\TrainClassPred.csv','w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        k=CTrain_P
        for i in k:
            writer.writerow([i])
            
    with open(DataDump+r'\TrainClassTrue.csv','w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        k=CTrain_T
        for i in k:
            writer.writerow([i])
    
    with open(DataDump+r'\TestClassPred.csv','w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        k=CTest_P
        for i in k:
            writer.writerow([i])
            
    with open(DataDump+r'\TestClassTrue.csv','w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        k=CTest_T
        for i in k:
            writer.writerow([i])