import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import numpy as np
from scipy.stats import norm
from numpy import linalg as la 
import tensorflow as tf
import tflearn
from matplotlib.ticker import FormatStrFormatter

bw=40e6
fres=100e3
linebw=5e6
linesperchunk=1000
flines= linesperchunk*4
nbins=int(bw/fres)
skiprows = int(bw/linebw)
patchtype = "random"
siglist={0:(800,1070),
         1:(1090,1155),
         2:(1170,1400),
         3:(1660,1720),
         4:(1960,2080),
         5:(2125,2175),
         6:(2200,2275),
         7:(3880,3960),
         8:(4220,4270),
         #9:(4750,4875),
         9:(6400,6600),
         #11:(6650,6800),
         10:(7900,8000),
         #13:(8000,8100),
         #14:(8100,8200),
         11:(9200,9600),
        }
'''
siglist={0:(9200,9600),
        }
'''


def get_patches(shape, psize, count=100):
    xpts = np.random.randint(shape[0]-psize[0],size=count)
    ypts = np.random.randint(shape[1]-psize[1],size=count)
    return (xpts,ypts)


def lnorm(X_train):
    print("Pad:", X_train.shape)
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            X_train[i,j,:] = np.nan_to_num(X_train[i,j,:]/la.norm(X_train[i,j,:],2))
    return X_train


def gendata(infile):
    farr = np.zeros((1,nbins))
    cntr=0
    for data in pd.read_csv(infile, header=None, chunksize=linesperchunk*skiprows):
        for i in range(linesperchunk): 
            sf = data[(skiprows*i):(skiprows*(i+1))].sort_values(by=2)
            dta= sf.as_matrix(range(6,int(linebw/fres)+6))
            farr=np.vstack((farr,dta.flatten()[:nbins]))
        cntr=cntr+linesperchunk
        if cntr>=flines:
            break
        print(cntr)
    farr=np.delete(farr,0,0)
    if patchtype == "random":
        tlen = 10 
        freqlen = 399 
        patchcnt = 8000
        pxorg,pyorg = get_patches(farr.shape,(tlen,freqlen),patchcnt)
        pyorg=np.zeros(patchcnt)
        patches=[]
        for i in range(patchcnt):
            patches.append(farr[pxorg[i]:pxorg[i]+tlen,:])
        
        patches=np.array(patches)
        print(patches.shape)
        train_data=patches
        train_data_org = np.copy(train_data)
        nmin = np.min(train_data) 
        #train_data = -np.min(train_data)+ train_data
        nmax = np.max(train_data) 
        #rain_data = (train_data - np.min(train_data))/(np.max(train_data) - np.min(train_data))
        #print "before:",np.mean(train_data), np.std(train_data)
        train_data = (train_data-np.mean(train_data))/np.std(train_data)
        #print "after:",np.min(train_data), np.max(train_data)

        train_labels = np.zeros((train_data.shape[0],1))
        return train_data,train_labels, nmin, nmax, train_data_org
    elif patchtype =="siglist":
        nsamples = max([el[1]-el[0] for k,el in siglist.items()])
        tsamples = 10 
        train_data = np.zeros((1,tsamples,nsamples))
        train_labels = np.zeros((1,len(siglist)))
        minval = np.min(farr)
        for key,el in siglist.items():
            dta = farr[:,el[0]:el[1]]
            res = np.zeros((farr.shape[0],nsamples)) + minval 
            #append zeros
            sigbw = el[1]-el[0]
            shift = sigbw/2
            mid = nsamples/2
            res[:,(mid-shift):(mid+sigbw-shift)] = dta
            maxidx = int(dta.shape[0]/tsamples)*tsamples
            res = res[:maxidx]
            train_cnt = res.shape[0]/tsamples
            if train_cnt > 1:
                train_data   = np.vstack((train_data,np.reshape(res,(train_cnt,tsamples,nsamples))))
                dummy_labels = np.zeros((train_cnt,len(siglist)))
                dummy_labels[:, key] = 1
                train_labels = np.vstack((train_labels,dummy_labels))
                print("Training data: Generation done for:", key)
        train_data = np.delete(train_data,0,0)
        train_labels = np.delete(train_labels,0,0)
        #print np.min(train_data)
        train_data_org = np.copy(train_data)
        nmin = np.min(train_data) 
        nmax = np.max(train_data) 
        #rain_data = (train_data - np.min(train_data))/(np.max(train_data) - np.min(train_data))
        train_data = (train_data-np.mean(train_data))/np.std(train_data)
        print("Min, max:",np.min(train_data), np.max(train_data))

        return train_data,train_labels, nmin, nmax, train_data_org

