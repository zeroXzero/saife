import numpy as np
import matplotlib.pyplot as plt
nsamples = 64 
train_cnt = 3000
maxbw=30
minbw=7
fcnt=6
scnt = nsamples * fcnt
predict=False
#ltraffic=["single_cont", "mult_cont", "single_rshort", "mult_rshort", "det_hop"]
ltraffic=["single_cont", "single_rshort", "mult_cont", "det_hop"]
#ltraffic=["single_cont", "single_rshort", "det_hop"]
dl=[]
for el in ltraffic:
    dl += [el]+["dummy"]*2
#ltraffic=dl
noiseval = 0.01
mindb=5
maxdb=20
plotenable=False

datatype = "float32"
train_data = np.zeros((1,fcnt,nsamples),dtype=datatype)
train_labels = np.zeros((1,len(ltraffic)),dtype=datatype)
bw_labels = [] 
pos_labels = [] 
nsig_labels = []

def gaussian(cnt):
    indata = noiseval/np.sqrt(2)* (np.random.normal(size=(cnt,fcnt,nsamples)) + 1j* np.random.normal(size=(cnt,fcnt,nsamples)))
    outdata = 20*np.log10(np.abs(np.fft.fft(indata)/float(nsamples)))
    outdata = np.fft.fftshift(outdata,axes=(2,))
    #outdata = -np.min(outdata)+outdata
    #outdata = (outdata- np.min(outdata))/(1 - np.min(outdata))
    return outdata

if predict:
	pred_data = np.zeros((1,pcnt,nsamples),dtype=datatype)

def gendata(noise=True, normalize=True):
    global train_data, train_labels, bw_labels, pos_labels, nsig_labels
    idx=0
    for traffic in ltraffic:
        if traffic == "single_cont":
            for i in range(train_cnt):
                sigbw = np.random.randint(minbw,maxbw)
                sigpos = np.random.randint(0, nsamples-sigbw)
                bw_labels.append((sigbw/float(nsamples)))
                pos_labels.append((sigpos+(sigbw/2.0))/float(nsamples))
                nsig_labels.append(1)
                sigon1=np.concatenate((np.ones((fcnt,sigbw)),np.zeros((fcnt,nsamples-sigbw))),axis=1)
                sigon1 = np.roll(sigon1,sigpos,axis=1).reshape(1,fcnt,nsamples)
                snr = np.random.uniform(mindb,maxdb)
                rintens = 10**(snr/20.0) * noiseval * nsamples/sigbw 
                sigon1 =  sigon1 * rintens #+ gaussian() 
                train_data   = np.vstack((train_data,sigon1))
            dlabels=np.zeros((train_cnt,len(ltraffic)))
            dlabels[:,idx]=1
            train_labels = np.vstack((train_labels,dlabels))
            if plotenable:
                plt.figure(1)
                print pos_labels[0]*nsamples
                print bw_labels[0]*nsamples
                plt.imshow(sigon1[0], interpolation='none')
                plt.colorbar(orientation='vertical')
                plt.show()
        elif traffic == "mult_cont":
            for i in range(train_cnt):
                nsig = np.random.randint(2,5)
                actsig=np.zeros((fcnt,nsamples))
                posum=0
                for i in range(nsig):
                    sigbw = np.random.randint(minbw,maxbw/2)
                    sigpos = np.random.randint(0, nsamples-sigbw)
                    posum += sigpos+(sigbw/2.0)
                    sigon1=np.concatenate((np.ones((fcnt,sigbw)),np.zeros((fcnt,nsamples-sigbw))),axis=1)
                    sigon1 = np.roll(sigon1,sigpos,axis=1).reshape(1,fcnt,nsamples)
                    snr = np.random.uniform(mindb,maxdb)
                    rintens = 10**(snr/20.0) * noiseval * nsamples/sigbw 
                    sigon1 =  sigon1 * rintens 
                    actsig = actsig + sigon1
                orgbw = len(np.where(actsig[0]>0)[0])
                bw_labels.append((orgbw/float(nsamples)))
                pos_labels.append(posum/float(nsamples)/nsig)
                nsig_labels.append(nsig)
                actsig = actsig #+ gaussian()  
                train_data   = np.vstack((train_data,actsig))
            dlabels=np.zeros((train_cnt,len(ltraffic)))
            dlabels[:,idx]=1
            train_labels = np.vstack((train_labels,dlabels))
            if plotenable:
                plt.figure(1)
                plt.imshow(actsig[0], interpolation='none')
                plt.colorbar(orientation='vertical')
                plt.show()
        elif traffic == "single_rshort":
            for i in range(train_cnt):
                sigbw = np.random.randint(minbw,maxbw)
                sigpos = np.random.randint(0, nsamples-sigbw)
                bw_labels.append((sigbw/float(nsamples)))
                pos_labels.append((sigpos+(sigbw/2.0))/float(nsamples))
                nsig_labels.append(1)
                ipres = np.random.randint(2,size=fcnt).reshape(fcnt,1)
                sig = np.repeat(ipres,sigbw,axis=1)
                sigon1=np.zeros((fcnt,nsamples))
                sigon1[:,sigpos:sigpos+sigbw] = sig
                snr = np.random.uniform(mindb,maxdb)
                rintens = 10**(snr/20.0) * noiseval * nsamples/sigbw 
                sigon1 =  sigon1 * rintens #+ gaussian()  
                sigon1= sigon1.reshape(1,fcnt,nsamples)
                train_data   = np.vstack((train_data,sigon1))
            dlabels=np.zeros((train_cnt,len(ltraffic)))
            dlabels[:,idx]=1
            train_labels = np.vstack((train_labels,dlabels))
            if plotenable:
                plt.figure(1)
                plt.imshow(sigon1[0], interpolation='none')
                plt.colorbar(orientation='vertical')
                plt.show()
        elif traffic == "mult_rshort":
            for i in range(train_cnt):
                nsig = np.random.randint(2,5)
                actsig=np.zeros((fcnt,nsamples))
                posum=0
                for i in range(nsig):
                    sigbw = np.random.randint(minbw,maxbw/2)
                    sigpos = np.random.randint(0, nsamples-sigbw)
                    posum += sigpos+(sigbw/2.0)
                    ipres = np.random.randint(2,size=fcnt).reshape(fcnt,1)
                    sig = np.repeat(ipres,sigbw,axis=1)
                    sigon1=np.zeros((fcnt,nsamples))
                    sigon1[:,sigpos:sigpos+sigbw] = sig
                    snr = np.random.uniform(mindb,maxdb)
                    rintens = 10**(snr/20.0) * noiseval * nsamples/sigbw 
                    sigon1 =  sigon1 * rintens 
                    actsig = actsig + sigon1
                actsig = actsig.reshape(1,fcnt,nsamples)
                #maxpos = np.argmax(np.sum(actsig>0,axis=0))
                #orgbw = len(np.where(actsig[0][maxpos]>0)[0])
                orgbw = np.max(np.sum(actsig>0,axis=0))
                bw_labels.append((orgbw/float(nsamples)))
                pos_labels.append(posum/float(nsamples)/nsig)
                nsig_labels.append(nsig)
                actsig = actsig #+ gaussian()  
                train_data   = np.vstack((train_data,actsig))
            dlabels=np.zeros((train_cnt,len(ltraffic)))
            dlabels[:,idx]=1
            train_labels = np.vstack((train_labels,dlabels))
            if plotenable:
                plt.figure(1)
                plt.imshow(actsig[0], interpolation='none')
                plt.colorbar(orientation='vertical')
                plt.show()
        elif traffic == "det_hop":
            for i in range(train_cnt):
                sigbw = np.random.randint(minbw,maxbw/2)
                sigpos = np.random.randint(0, nsamples-sigbw)
                detshift = np.random.randint(1,maxbw/4)
                initarr=np.zeros(nsamples)
                initarr[sigpos:sigpos+sigbw] = 1 
                res=[]
                for j in range(fcnt):
                    res.append(initarr)
                    initarr= np.roll(initarr,detshift)
                res = np.reshape(res,(1,fcnt,nsamples))
                #orgbw = sigbw+detshift*fcnt
                orgbw = sigbw
                bw_labels.append((orgbw/float(nsamples)))
                pos_labels.append((sigpos+(sigbw/2.0)+(fcnt*detshift/2))%nsamples/float(nsamples))
                snr = np.random.uniform(mindb,maxdb)
                rintens = 10**(snr/20.0) * noiseval * nsamples/sigbw 
                res = res * rintens #+ gaussian()  
                train_data = np.vstack((train_data,res))
                nsig_labels.append(1)
            dlabels=np.zeros((train_cnt,len(ltraffic)))
            dlabels[:,idx]=1
            train_labels = np.vstack((train_labels,dlabels))
            if plotenable:
                plt.figure(1)
                plt.imshow(res[0], interpolation='none')
                plt.colorbar(orientation='vertical')
                plt.show()
        else:
            print("Traffic not defined")
        idx+=1
    
    train_data = np.delete(train_data,0,0)
    train_labels = np.delete(train_labels,0,0)
    bw_labels = np.reshape(bw_labels,(-1,1))
    pos_labels = np.reshape(pos_labels,(-1,1))
    nsig_labels = np.reshape(nsig_labels,(-1,1))
    if predict:
        pred_data = np.delete(pred_data,0,0)
        print("Predict data:",pred_data.shape)
    print("--"*50)
    print("Training data:",train_data.shape)
    print("--"*50)
    #print len(train_data), len(train_labels), len(bw_labels), len(pos_labels)
    if noise:
        #train_data_nn = train_data
        #train_data = train_data+gaussian(train_data.shape[0])
        train_data = train_data+ noiseval
    train_data_org = np.copy(train_data)
    train_data = 20 * np.log10(train_data)
    nmin = np.min(train_data) 
    train_data = -np.min(train_data)+train_data
    nmax = np.max(train_data) 
    train_data = (train_data- np.min(train_data))/(np.max(train_data) - np.min(train_data))
    return train_data, train_labels, bw_labels, pos_labels, nmin, nmax,  train_data_org

