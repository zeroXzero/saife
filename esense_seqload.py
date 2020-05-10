import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import numpy as np
from scipy.stats import norm
from numpy import linalg as la 
from collections import OrderedDict
import tensorflow as tf
import tflearn
import os
import operator
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter


def lnorm(X_train):
    print "Pad:", X_train.shape
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            X_train[i,j,:] = np.nan_to_num(X_train[i,j,:]/la.norm(X_train[i,j,:],2))
    return X_train

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def save_images(res,name):
    plt.figure()
    plt.imshow(res, interpolation='none', aspect='auto')
    plt.colorbar(orientation='vertical')
    plt.savefig(name)
    plt.close()

def gendata(mydir,hrenable=True,testenable=False):
    hrenable = True 
    testenable= True
    trainhrs=24*7
    files = []
    for file in os.listdir(mydir):
        if file.endswith(".npy"):
                files.append(os.path.join(mydir, file))
    
    labels={}
    lfiles={}
    count=0
    for f in files:
        fname = f.split("/")[-1]
        if not labels.has_key(fname.split("_")[0]):
            labels[fname.split("_")[0]]=count
            lfiles[fname.split("_")[0]]=[]
            count+=1
        if hrenable:
            if testenable:
                if int(fname.split("_")[1].split(".")[0]) > trainhrs:
                    lfiles[fname.split("_")[0]].append(f)
            else:
                if int(fname.split("_")[1].split(".")[0]) < trainhrs:
                    lfiles[fname.split("_")[0]].append(f)
        else:
            lfiles[fname.split("_")[0]].append(f)
    
    
    labels = OrderedDict(sorted(labels.items(), key=operator.itemgetter(1)))
    
    
    for key in lfiles.keys():
        lfiles[key].sort(key = lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0]))
    
    num_labels = len(labels)
    nsamples = 0
    tsamples = 6 
    mods=labels
    
    for f in files:
        dta = np.load(f)
        if len(dta.shape)>1:
            if nsamples < dta.shape[1]:
                nsamples = dta.shape[1]
    
    
    datatype = "float32"
    train_data = np.zeros((1,tsamples,nsamples),dtype=datatype)
    test_data =  np.zeros((1,tsamples,nsamples),dtype=datatype)
    train_labels = np.zeros(num_labels)
    test_labels = np.zeros(num_labels)

    print("--"*50)
    for key in labels.keys():
        print key
        for f in lfiles[key][:24*20]:
            dta = np.load(f)
            if len(dta.shape)<2:
                continue
            res = np.zeros((dta.shape[0],nsamples))
            #append zeros
            shift = dta.shape[1]/2
            mid = nsamples/2
            res[:,(mid-shift):(mid+dta.shape[1]-shift)] = dta
            maxidx = int(dta.shape[0]/tsamples)*tsamples
            res = res[:maxidx]
            if np.isnan(res).any():
                print "NAN found",f
                continue
            train_cnt = res.shape[0]/tsamples
            if train_cnt > 1:
                #save_images(lnorm(np.reshape(res,(train_cnt,tsamples,nsamples)))[0],sample_directory+'/fig'+key+'.png')
                resdta = np.reshape(res,(train_cnt,tsamples,nsamples))
                #resdta = (resdta- np.min(resdta))/(np.max(resdta) - np.min(resdta))
                train_data   = np.vstack((train_data,resdta))
                dummy_labels = np.zeros((train_cnt,len(labels)))
                dummy_labels[:, labels[key]] = 1
                train_labels = np.vstack((train_labels,dummy_labels))
                #print("Training data: Generation done for:", key)
    train_data = np.delete(train_data,0,0)
    train_labels = np.delete(train_labels,0,0)
    train_data_org = np.copy(train_data)
    Y_train = train_labels
    
    #train_data = lnorm(train_data)
    #train_data = -np.min(train_data)+train_data
    #train_data = (train_data- np.min(train_data))/(np.max(train_data) - np.min(train_data))
    print("Before:",np.mean(train_data), np.std(train_data))
    train_data = (train_data-np.mean(train_data))/np.std(train_data)
    print ("After:",np.min(train_data), np.max(train_data))
    
    #out0 = (out0-np.mean(out0))/np.std(out0)
    
    X_train = np.reshape(train_data,(-1,tsamples,nsamples))
    
    return X_train,Y_train,labels,train_data_org


#gendata("/xtra/esense_data_jan2019/202481588271454_1541682442")

