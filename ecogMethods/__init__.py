#!/usr/bin/env python
# coding: utf-8

# In[7]:


from __future__ import with_statement
from __future__ import division

import sys
import numpy as np
import pandas as pd
import wave
import os
import os.path as op
import h5py
import math
import warnings
import time
import scipy.signal as sig
import hdf5storage

import scipy as sp

from scipy.signal import spectrogram, hamming, resample_poly
from scipy.io import wavfile as wf

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
plt.ion()
plt.style.use('seaborn-white')

from sklearn.linear_model import Ridge, Lasso, SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_score

get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib qt


# Classes

# In[30]:


class Events:
    def __init__(self):
        self.event = []
        self.onset = []
        self.badevent = []
        self.offset = []
        self.block =[]
        self.react = []


# In[31]:


class Globals:
    def __init__(self, SJdir, ANdir, DTdir, subj, srate, ANsrate, elecs, bad_elecs, tank):
        self.SJdir= SJdir
        self.ANdir= ANdir
        self.DTdir= DTdir
        self.subj= subj
        self.srate= srate 
        self.ANsrate= ANsrate 
        self.elecs= elecs 
        self.bad_elecs= bad_elecs
        self.tank= tank


# In[32]:


## whats parameter g? could reduce number of params/attributes
class Params:
    def __init__(self, st, en, plot, baseline, bl_st, bl_en, scale, noCar, g):
        self.st   = -200            #start time window
        self.en   = 1700            #end time window
        self.plot = 200             #plotting x-axis ticks every 200 ms
        self.baseline = True          #baseline flag
        self.bl_st    = -250        #baseline start 
        self.bl_en    = -50         #baseline end
        self.scale = 0.7            #scale colorbar to [-0.8 0.8]*maximal response
        self.noCar = False
        self.ignore_target= ''
        self.limit=''
        self.thickness=''
        self.gauss_width=''
        self.sort=''
        self.do_plot=''
        self.swap=''
        self.pseudo=''
        self.response_lock=False
        self.shade_plot = True
        


# In[ ]:





# In[ ]:





# In[ ]:





# Methods

# In[33]:


def save_h5(path, name, data): #save HDF5 files
    hf = h5py.File(path, "w")
    hf.create_dataset(name, data = data)
    hf.close()


# In[34]:


def load_h5(path, name): # load HDF5 files
    hf = h5py.File(path, "r")
    z = hf.get(name)
    z = z[...]
    #if is_globals
    hf.close()
    return z


# In[35]:


def create_subj_globals(subj, block, srate, ANsrate, elecs, bad_elecs, TANK, 
                        root_path = r'\\research-cifs.nyumc.org\Research\Epilepsy_ECOG\SharedAnalysis\Testing',
                       create_dir= True, NY=False):
    a = "analysis"
    d = "data"
    
    direcEr = 'Directory not created' # messages shown to user
    creatD = 'Created directory'
    
    pre_dir = "" #pre-diret
    if sys.platform.startswith("lin"):
        pre_dir = "~/"
        
    if create_dir== True:
        if root_path.endswith(subj):
            pth = root_path
            print("Continuing in current directory", root_path)
        elif root_path.endswith("subjsPython"):
            pth = op.join(root_path, subj)
        else:
            pth = op.join(root_path, "subjsPython", subj)
    #             pth2 = op.join(root_path, "subjsPython")
    #             if not op.exists(pth2):
    #                 os.makedirs(pth2)
    #                 if op.exists(pth2):
    #                     print (creatD, pth2) 
    #                 else: raise Exception(direcEr)

        if not op.exists(pth):
            os.makedirs(pth)
            if op.exists(pth):
                print (creatD, pth)
            else: raise Exception(direcEr)
    else:
        pth = root_path
        
    if ~subj.startswith("NY") & NY==True:
        raise ValueError("Subject number must start with NY")
        
    SJdir = op.join(pre_dir, pth)
    SJdirA = op.join(SJdir, a)
    SJdirD = op.join(SJdir, d)

    ANdir = op.join(SJdirA, block)
    DTdir = op.join(SJdirD, block)
    
    if not op.exists(SJdir):
        raise ValueError(f'Directory {SJdir} does not exist, please create it')
    
    if not op.exists(SJdirA):
        os.makedirs(SJdirA)
        if op.exists(SJdirA):
            print (creatD, SJdirA) 
        else: raise Exception(direcEr)
            
    if not op.exists(ANdir):
        os.makedirs(ANdir)
        if op.exists(ANdir):
            print (creatD, ANdir) 
        else: raise Exception(direcEr)
            
    if not op.exists(SJdirD):
        os.makedirs(SJdirD)
        if op.exists(SJdirD):
            print (creatD, SJdirD) 
        else: raise Exception(direcEr)
            
    if not op.exists(DTdir):
        os.makedirs(DTdir)
        if op.exists(DTdir):
            print (creatD, DTdir) 
        else: raise Exception(direcEr) 
            
    if not 'tank' in locals(): tank= []
    
    SG = op.join(ANdir, "subj_globals.h5")
    print('Saving global variables in ' + str(SG))
    
    if op.exists(SG):
        os.remove(SG)
        
    g = Globals(SJdir, ANdir, DTdir, subj, srate, ANsrate, elecs, bad_elecs, tank)
    
    dt = h5py.special_dtype(vlen=bytes)
    hf = h5py.File(SG, 'w')
    grp = hf.create_group('subject_globals')
    gdirs = np.array([g.SJdir, g.ANdir, g.DTdir, g.subj, str(print(g.tank))])

    # creating H5 datasets
    asciiList = [n.encode("ascii", "ignore") for n in gdirs]

    dirSet = grp.create_dataset("gdirs",(len(asciiList),1), dtype='S100', data=asciiList)
    srateSet = grp.create_dataset("srate", dtype=int, data=g.srate)
    ansrateSet = grp.create_dataset("ANsrate", dtype=int, data=g.ANsrate)
    elecSet = grp.create_dataset("elecs", (np.asarray(np.shape(g.elecs))), dtype=int, data= np.asarray(g.elecs))
    badSet = grp.create_dataset("bad_elecs", (np.asarray(np.shape(g.bad_elecs))), dtype=int, data= np.asarray(g.bad_elecs))
    hf.close()
    


# In[9]:


def get_subj_globals(subj, block, root_path = r'\\research-cifs.nyumc.org\Research\Epilepsy_ECOG\SharedAnalysis\Testing',
                    create_dir= True, NY=False, from_mat=False, matDir=""):
    a = "analysis"
    if from_mat==True:
        glob= hdf5storage.loadmat(op.join(matDir,a,block,'subj_globals.mat'))
        G = Globals(op.join(matDir),op.join(matDir,a,block), op.join(matDir,'data',block), glob['subj'][0],glob['srate'][0][0],
                    glob['ANsrate'][0][0], glob['elecs'][0], glob['bad_elecs'][0], glob['TANK'][0])
    else:
        # would need to comment out if sharing with others
        if create_dir==True:
            if root_path.endswith(subj):
                pth = root_path
                print("Continuing in current directory", root_path)
            elif root_path.endswith("subjsPython"):
                pth = op.join(root_path, subj)
            else:
                pth = op.join(root_path, "subjsPython", subj)
        else:
            pth = root_path
            print("Continuing in current directory", root_path)
        if ~subj.startswith("NY") & NY==True:
            raise ValueError("Subject number must start with NY")
        pre_dir = ""
        if sys.platform.startswith("lin"):
            pre_dir = "~/"
        globals_dir = op.join(pre_dir, pth, a, block, "subj_globals.h5")
        hf = h5py.File(globals_dir, "r")
        x = hf.get('subject_globals')
        dirs= x['/subject_globals/gdirs']
        dirs=dirs[...]
        gsrate= np.array(x['/subject_globals/srate'])
        gANsrate= np.array(x['/subject_globals/ANsrate'])
        gelecs= np.array(x['/subject_globals/elecs'])
        gbads= np.array(x['/subject_globals/bad_elecs'])
        G = Globals(dirs[0][0].decode("utf-8"),dirs[1][0].decode("utf-8"), dirs[2][0].decode("utf-8"),
                    dirs[3][0].decode("utf-8"), int(gsrate), int(gANsrate), gelecs, gbads, dirs[4][0].decode("utf-8"))
        hf.close()
    return G


# In[4]:


def extract_task_events(data, times, task, subj, srate = 512, start = 0, stop = '', eventMin = 150, thresh=''):
    scalar = int(srate/512)
    eventMin = eventMin*scalar
    if stop =='':
        stop==input("Need a value for 'stop'")
    
    data = data-np.mean(data)#mean center
    data = data/abs(max(data.T))#normalize
    data= data.clip(min=0)
    data = sig.savgol_filter(data,window_length= 93, polyorder=1 )
    if thresh=='':
        thresh= abs(max(data.T))/2
    i=0
    e1 = Events()
    e1.onset=[]
    e1.event=[]
    e1.badevent=[]
    onsets=0

    while i < len(times):

        if data.T[i]>thresh:
            e1.onset.append(round(times[i]*srate))
            onsets+=1
            e1.event.append(task+'_'+str(len(e1.onset)))
            i=i+(eventMin)
            if len(e1.onset)<4:
                e1.badevent.append(1)
            else:
                e1.badevent.append(0)
        i=i+1

    print('Found {0} events for {1} {2} \n'.format(len(e1.onset), subj, task))
    e1.onset = np.subtract(e1.onset,start)
    
    return e1


# In[10]:


def get_events(subj,block,pth, from_mat=False, matDir=""):
    
    eve=Events()
    
    if from_mat==False:
        a = 'analysis'
        pre_dir = ""
        if sys.platform.startswith("lin"):
            pre_dir = "~/"

        event_dir = op.join(pre_dir, pth, "events.h5")
        hf = h5py.File(event_dir, "r")
        z = hf.get('Events')
        onset=z['/Events/onset']
        onset=onset[...]
        event=z['/Events/event']
        event=event[...]
        badevent=z['/Events/badevent']
        badevent=badevent[...]
        
        eve.event=event 
        eve.onset=onset 
        eve.badevent=badevent
        hf.close()
        
    else:
        
        matEvents=hdf5storage.loadmat(op.join(matDir,'analysis',block,'events.mat'))
        for i in matEvents['Events'][0]:
            eve.event.append(i[0][0])
            eve.onset.append(i[1][0][0])
            eve.offset.append(i[2][0][0])
            eve.badevent.append(i[3][0][0])
            try:
                eve.stimfile.append(i[4][0][0][0])
                eve.stim.append(i[5][0][0][0])
                eve.responset.append(i[6][0][0])
                eve.respoffset.append(i[7][0][0])
                eve.variable8.append(i[8][0][0])
                eve.variable9.append(i[9][0][0])
            except:
                continue
    return eve


# In[39]:


def band_pass(signal, sampling_rate=1000, lower_bound=70, upper_bound=150, tm_OR_fr= 1, wind= 'flatgauss'):
    #       signal        - input signal to be filtered (time or frequency domain)
    #       sampling_rate - signal's sampling rate
    #       lower_bound   - lower frequency bound for bandpass filtering
    #       upper_bound   - upper frequency bound for bandpass filtering
    #       tm_OR_fr      - 1 if the input signal is in the time domain, 0 if it
    #                       is in the frequency domain
    #       wind          - windows type:
    #                         'HMFWgauss' - HMFW of upper_bound-lower_bound
    #                         'flatgauss' - gaussian with a the maximum point
    #                                       flatened to upper_bound-lower_bound
    #                                       length

    #    The function returns the filtered hilbert signal (low->high) in the time domain
    
    max_freq=sampling_rate/2
    df=sampling_rate/len(signal)
    center_freq=(upper_bound+lower_bound)/2
    filter_width=upper_bound-lower_bound
    x= np.arange(0,max_freq+1,df)
    gauss_width = 1
    
    if wind != 'flatgauss' and wind!= 'HMFWgauss':
        gauss_width = wind
        wind = 'flatgauss'

    if wind=='flatgauss':
        gauss=np.exp(-1*gauss_width*((x-center_freq)**2)) 
        cnt_gauss = round(center_freq/df) 
        flat_padd = round(filter_width/df)   # flat padding at the max value of the gaussian
        padd_left = np.floor(flat_padd/2) 
        padd_right = np.ceil(flat_padd/2) 
        our_wind=np.append(gauss[int(padd_left):int(cnt_gauss+1)],np.ones(int(flat_padd)))
        our_wind=np.append(our_wind,gauss[int(cnt_gauss+1):int(len(gauss)-padd_right)])
        
    elif wind=='HMFWgauss':
        sigma = np.true_divide(filter_width,2*sqrt(2*mt.log(2)))          # standrad deviation to conform with HMFW of filter_width
        gauss= np.true_divide(np.exp((-1*(x-center_freq)**2)),2*sigma**2)
        our_wind = gauss
        
    else:
        raise ValueError("No valid window")
    
    if lower_bound == 0:
        our_wind[0:round(centre_freq/df)] = iter([1 for i in range(len(round(centre_freq/df)))])
            
    if len(signal)%2 == 0:
        our_wind = our_wind[:-1]
    our_wind = np.append(our_wind, our_wind[::-1])
        
    if tm_OR_fr==1:
        signal=np.fft.fft(signal, axis = 0)
    
    windowed_signal = signal*our_wind
    L = int(np.shape(signal)[0]/2 + 1)
    
    filt_signal = np.fft.irfft(windowed_signal[0:L], axis = 0)
    
    return filt_signal


# In[40]:


def my_hilbert(signal, sampling_rate=1000, lower_bound=70, upper_bound=150, tm_OR_fr= 1, wind= 'flatgauss'):
    #       signal        - input signal to be filtered (time or frequency domain)
    #       sampling_rate - signal's sampling rate
    #       lower_bound   - lower frequency bound for bandpass filtering
    #       upper_bound   - upper frequency bound for bandpass filtering
    #       tm_OR_fr      - 1 if the input signal is in the time domain, 0 if it
    #                       is in the frequency domain
    #       wind          - windows type:
    #                         'HMFWgauss' - HMFW of upper_bound-lower_bound
    #                         'flatgauss' - gaussian with a the maximum point
    #                                       flatened to upper_bound-lower_bound
    #                                       length

    #    The function returns the filtered hilbert signal (low->high) in the time domain
    
    
    max_freq=sampling_rate/2
    df=sampling_rate/len(signal)
    center_freq=(upper_bound+lower_bound)/2
    filter_width=upper_bound-lower_bound
    x= np.arange(0,max_freq+1,df)
    gauss_width = 1
    
    if wind != 'flatgauss' and wind!= 'HMFWgauss':
        gauss_width = wind
        wind = 'flatgauss'
        
    if wind=='flatgauss':
        gauss=np.exp(-1*gauss_width*((x-center_freq)**2)) 
        cnt_gauss = round(center_freq/df) 
        flat_padd = round(filter_width/df)   # flat padding at the max value of the gaussian
        padd_left = np.floor(flat_padd/2) 
        padd_right = np.ceil(flat_padd/2) 
    
        our_wind=np.append(gauss[int(padd_left):int(cnt_gauss+1)],np.ones(int(flat_padd)))
        our_wind=np.append(our_wind,gauss[int(cnt_gauss+1):int(len(gauss)-padd_right)])
    elif wind=='HMFWgauss':
        sigma = np.true_divide(filter_width,2*sqrt(2*mt.log(2)))          # standard deviation to conform with HMFW of filter_width
        gauss= np.true_divide(np.exp((-1*(x-center_freq)**2)),2*sigma**2)
        our_wind = gauss
    else:
        raise ValueError("No valid window")
        
    # zero pad    
    our_wind=np.append(our_wind, np.zeros(len(signal)-len(our_wind)))
    
    if tm_OR_fr==1:
        signal=np.fft.fft(signal)

    our_wind[0] = our_wind[0]/2; # DC component is halved
    our_wind = 2*our_wind

    filt_signal = np.fft.ifft(signal*our_wind)

    return filt_signal


# In[41]:


def my_conv(data = np.array([]), length = 100):
    win= np.hanning(length)
    smooth=[]
    if np.shape(data)[1] == 1:
        data=data.T
    if data=='' or data==[] or data==np.array([]):
        return smooth
    if length=='':
        smooth=data
        return smooth
    for i in np.arange(np.shape(data)[0]):
        smooth.append(np.divide(np.convolve(data[i],win, 'same'),sum(win)))#np.convolve(data,(i,:),win(length), 'same')
    return smooth


# In[42]:


def create_CAR(subj, block, bad_elecs, root_path, create_dir= True, NY=False):
    
    
    if create_dir==True:
        if root_path.endswith(subj):
            data_path = op.join(root_path, "data", block, "gdat.h5")
        elif root_path.endswith("subjsPython"):

            data_path = op.join(root_path, subj, "data", block, "gdat.h5")
        else:
            data_path = op.join(root_path, "subjsPython", subj, "data", block, "gdat.h5")
    else:
        data_path = op.join(root_path, "data", block, "gdat.h5")
        
    if ~subj.startswith("NY") & NY==True:
        raise ValueError("Subject number must start with NY")
        
    data = load_h5(data_path, "gdat")
    good_data = [data[i] for i in range(len(data)) if i not in bad_elecs]
    good_data_zero_meaned = [good_data[i] - np.mean(good_data[i]) for i in range(len(good_data))]
    reference = np.mean(good_data_zero_meaned, axis = 0)
    reference.resize((1,np.shape(reference)[0]))
    
    data_zero_meaned = [data[i] - np.mean(data[i]) for i in range(len(data))]
    
    new_data = np.subtract(data_zero_meaned, reference)
    
#     if root_path.endswith(subj):
            
    save_h5(op.join(data_path[:-8],'car.h5'), "car", reference)

#    print('saving CAR')
    save_h5(op.join(data_path[:-8], 'car_data.h5'), "car_data", new_data)
# #         print('saving Car Data')
#     elif root_path.endswith("subjsPython"):
#         save_h5(op.join(root_path, subj, "data", block,'car.h5'), "car", reference)
# #         print('saving CAR')
#         save_h5(op.join(root_path, subj, "data", block, 'car_data.h5'), "car_data", new_data)
# #         print('saving Car Data')
            
#     else:
#         save_h5(op.join(root_path, "subjsPython", subj, "data", block,'car.h5'), "car", reference)
# #         print('saving CAR')
#         save_h5(op.join(root_path, "subjsPython", subj, "data", block, 'car_data.h5'), "car_data", new_data)
# #         print('saving Car Data')
    print('saving car and reference')
    return new_data, reference


# In[63]:


def plot_single(subj, task, elec, params, root_path,
                f1=75, f2=150, raw=0, gdat = '', db=0, ignore_target='', from_mat=False, matDir=''):
#     raw    - 0 for raw trace, 1 for power envelope
#     db     - flag to go into debug mode after plotting
#     params - default are:
#        params.st   = -200;            #start time window
#        params.en   = 1700;            #end   timw window
#        params.plot = 200;             #plotting x-axis ticks every 200 ms
#        params.baseline = 1;           #baseline flag
#        params.bl_st    = -250;        #baseline start
#        params.bl_en    = -50;         #baseline end
#        params.scale = 0.8;            #scale colorbar to [-0.8 0.8]*maximal response
#
#  Usage:
#        plot_single_phr('JH1','phr',45,70,150);
#        plot_single_phr('JH1','phr',45,0.1,20,1);
#
#        params.st   = -500;
#        params.en   = 2000;
#        params.plot = 250;
#        params.baseline = 0;
#        params.scale = 0.8;
#        plot_single('JH2','phr',22,0.1,20,0,params)
    TrialsMTX = [] ############################# was defined as params, we don't know why
    x = get_subj_globals(subj, task, root_path, from_mat=from_mat, matDir=matDir)
    if gdat == '':
        if from_mat==True:
            if params.noCar == False:
                gdat = hdf5storage.loadmat(op.join(x.DTdir,"gdat_CAR.mat"))
                gdat=gdat['gdat']
            else:
                gdat = hdf5storage.loadmat(op.join(x.DTdir,"gdat.mat"))
                gdat=gdat['gdat']
        else:
            if params.noCar == False:
                gdat = load_h5(op.join(x.DTdir,"car_data.h5"),"car_data")
            else:
                gdat = load_h5(op.join(x.DTdir,"gdat.h5"), "gdat")
    events = get_events(x.subj, task,x.ANdir, from_mat=from_mat, matDir=matDir)
    if elec == "mic" or elec == "spkr":
        elec = load_h5(op.join(x.DTdir,"mic.h5"), "mic")
        exec("band = "+elec)
        elec = 1
        x.srate = x.ANsrate
    else:
        band = gdat[elec, :]
    if x.srate != 1000:
        band = sig.resample_poly(band, 1000, x.srate)
        srate = 1000
    pseudo = 0
    #pre = block
    thickness = 2
    gauss_width = 1
    srt = 1
    do_plot=1
    start_time_window = params.st
    end_time_window = params.en
    plot_jump = params.plot
    baseline = params.baseline
    if baseline:
        bl_st  = round(params.bl_st /1000*srate)
        bl_en  = round(params.bl_en /1000*srate)
    scale = params.scale
    if params.ignore_target != "":
        ignore_target = params.ignore_target
    if params.limit != "":
        limit = params.limit
    if params.thickness != "":
        thickness = params.thickness
    if params.gauss_width != "":
        gauss_width = params.gauss_width
    if params.sort != "":
        srt = params.sort
    if params.do_plot != "":
        do_plot = params.do_plot
    if params.swap != "" and params.swap:
        onsets = onsets_r;
    if params.pseudo != "":
        pseudo = params.pseudo;
    clr_res = 'k'
    clr_h   = 'k'
    tm_st  = round(start_time_window/1000*srate)
    tm_en  = round(end_time_window/1000*srate)
    events.onset=np.round(events.onset/x.srate*srate)
    jm = round(plot_jump/1000*srate)
    if raw == 0:
        band = abs(my_hilbert(band, srate, f1, f2))
#         print("plot mode - analytic amplitude")
    elif raw == 1:
        band = band_pass(band, srate, f1, f2)
#         print('plot mode - raw trace')
    elif raw == 2:
        band = band_pass(band, srate, f1, f2)
#         print("plot mode - raw zscores of baseline")
    elif raw == 3:
        band = abs(my_hilbert(band, srate, f1, f2))
#         print('plot mode - amplitude zscores of baseline')
    elif raw == 4:
        band = my_hilbert(abs(my_hilbert(band, srate, f1, f2)), srate, params.nest_f1, params.nest_f2)
        baseline = 0
        scale = 1
        clr_res = 'k'
#         print('plot mode - theta hilbert of amplitude')
    elif raw == 5:
        band = my_hilbert(band, srate, f1, f2, 1, 'HMFWgauss')
        baseline = 0
        scale = 1
        clr_res = 'k'
#         print('plot mode - hilbert')
    elif raw == 6:
        band =  20*np.log10(abs(my_hilbert(band, srate, f1, f2)))
        scale = 1
#         print('plot mode - log power compared to baseline')
    else:
        raise ValueError("raw values can only be 0 through 6")
    cnt = 0
    if ignore_target == '':
        ignore_target = 1
    MTXinds = []
    # probably want to loop through each Events object (one per task)
    # event called "onsets"
    for i in np.arange(len(events.event)):
        if events.badevent[i]:
            continue
        # istarget isnt an Events attribute
#         if ignore_target and (block == "word" or block == "ph") and event.istarget:
#             continue
        if params.limit != "":
            if cnt >= limit:
                continue
        cnt += 1
        if params.response_lock:
            tm_stmps = np.arange(events.onset_r[i]+tm_st, event.onset_r[i]+tm_en) #we never defined event_r
        else:
            tm_stmps = np.arange(events.onset[i]+tm_st, events.onset[i]+tm_en)## make sure this marks indices of trails in full signal
        tm_stmps=tm_stmps.astype(int) #convert to ints
        if baseline:
            bl_stmps = np.arange(events.onset[i]+bl_st, events.onset[i]+bl_en)
            bl_stmps=bl_stmps.astype(int)
            if raw == 2 or raw == 3:
                TrialsMTX = (band[tm_stmps] - np.mean(band[tm_stmps], 0))## not clear if it will run
            else:
                TrialsMTX.append(np.divide(100*(band[tm_stmps] - np.mean(band[bl_stmps],0)),np.mean(band[bl_stmps],0)))
        elif baseline == 2:
            bl_stmps = np.arange(events.event[i]+bl_st, events.event[i]+en_st)
            TrialsMTX = band[tm_stmps] - np.mean(band[tm_stmps], 0)
        else:
            TrialsMTX.append(band[tm_stmps])
    return TrialsMTX


# In[8]:


def detect_bads(signal, low_bound=10, up_bound=65, convolve = False, wind = 0, plot=False,thresh=1):
    datMaxs= [max(abs(i)) for i in signal]
    maxDev= np.std(datMaxs)
    maxZs= [(i-np.mean(datMaxs))/maxDev for i in datMaxs]
    maxBads= [i for i,val in enumerate(maxZs) if np.abs(val) >=.8*thresh]
    ft_sig = [np.abs(np.fft.fft(i))[low_bound:up_bound] for i in signal]
    datMeans= [np.mean(i) for i in ft_sig[:]]
    datDev= np.std(datMeans)
    datZs= [(i-np.mean(datMeans))/datDev for i in datMeans]
    zBads= [i for i,val in enumerate(datZs) if np.abs(val) >=thresh/2]
    
 
    return set(maxBads+zBads)


# In[3]:


def extract_blocks(data, times, subj, tasks=[], 
                   srate=512, blockMin=90000, eventMin=256,gap=2000, trigger_len=700, thresh=.09):
    scalar = int(srate/512)
    blockMin = scalar * blockMin
    eventMin = scalar * eventMin
    gap = scalar * gap
    trigger_len = scalar * trigger_len

    
    data = data-np.mean(data)#mean center
    data=data.clip(min=0)
    data= data/abs(max(data.T))#normalize
    task=['PicN','VisRead','AudRep','AudN','SenComp']
    i=0
    blocks=[]
    spikes=0
   
    while i< len(times):
        
        j=0
        if data.T[i]>thresh:
         
            spikes+=1
            hit=times[i]*srate # marks index-time for trigger 
            i=i+70*scalar       #advances more than length of trigger spike
            
            while j<50*scalar:  #searches for next spike of block-level trigger
                
                j=j+1
                if data.T[i+j]>thresh: # if found, mark the hit time
                    blocks.append(hit+trigger_len)
                    i=i+blockMin  #advance a little below the minimum block time
                    j=51*scalar         # exit j-loop
                    
                    #### need to account for finding task triggers
            i=i+50*scalar
    
        i=i+1
    sums=[]
    blocks=np.asarray(blocks)
    blocks=blocks.astype(int)
   
    tasks = ['VisRead','PicN','AudRep','AudN','SenComp'] 
    taskorder=[]
    for i in blocks:
         
        trigSum=sum(data[0][math.floor(i-trigger_len):i])
        
        sums.append(trigSum)
    
    def sortFirst(val): 
        return val[0] 
    sumEn=[(v,i) for i,v in enumerate(sums)]
    sumEn.sort(key=sortFirst, reverse=True)
   
    order=[v for i,v in sumEn]

    taskorder=[task[i] for i in order]
            
    print('Found {0} blocks for {1}\n'.format(len(blocks), subj))
    print('task order: ',taskorder)
    block_times=[]
    for t,v in enumerate(blocks):
        try:
            block_times.append([int(v+trigger_len),int(blocks[t+1]-gap)])
        except:
            if hit+gap <= len(data):
                block_times.append([int(v+trigger_len),int(hit+gap)])
                
            else:
                block_times.append([int(v+trigger_len),int(times[-1]*srate)])
                
        
#     print('Block order followed by index times are as follows: {0} \n {1}'.format(taskorder[:], block_times[:]))   
    print('Block index times are as follows: {0}'.format(block_times[:]))  
    return order, block_times


# # Signal processing methods

# In[46]:


# zero mean each feature and divide each feature by its L2 norm
def normalize(matrix):
    num_feats = np.shape(matrix)[0]
    the_mean = np.reshape(np.mean(matrix, axis = 1), (num_feats, 1))
    matrix = matrix - the_mean
    
    the_norm = np.reshape(np.linalg.norm(matrix, axis = 1), (num_feats,1))
    matrix = matrix/the_norm
    return matrix


# In[60]:


def time_delay_updated(X_data, wind_len):
    n_freq = np.shape(X_data)[1]
    # Make X and y one trial
    X_one_trial = np.concatenate(X_data, axis = 1)
    
    x = []
    x.append(X_one_trial[:, :]) #nondelayed collapsed spec
    
    for i in range(1, wind_len):
        the_pad = np.zeros((n_freq, i)) ## or voss i guess?
        the_data = np.concatenate((X_one_trial[:, i:], the_pad) , axis = 1) #pad our delayed data
        x.append(the_data)
    
    x = np.concatenate(x, axis = 0) #collapse again so each row (freq) is trials x time delays
    return np.asarray(x)


# In[58]:


# Pad signal with noise, or otherwise zeros
# aud_len and len_spec_window in seconds
def pad_data(signal, sf, aud_len, len_spec_window, noise = True):
    
    total_length = int(sf*(aud_len + len_spec_window))
    pad_length = total_length - len(signal)
    
    pad = voss(pad_length) if noise else np.zeros(pad_length)
        
    new_signal = np.concatenate((signal, pad))
    return new_signal


# In[47]:


def file_to_waveform(file, make_plot = False, return_info = False):
    
    with wave.open(file,'r') as wav_file:
        # Data from audio file stored as an np.array with tuples for each frame
        # Each component of tuple represents one of two audio channels
        fs, signal = wf.read(file)
        
        # Grab audio data from first audio channel if its a stereo signal
        if wav_file.getnchannels() == 2:
            signal = signal[:,0]
        
        # Number of frames in channel
        frames = len(signal)
        
        # Length of audio file
        seconds = frames/fs
        
        # Creates interval spanning channel with time instance at each frame 
        Time = np.linspace(0, seconds, num = frames)
        
        # Makes plot if you want
        if make_plot:
            plt.title('Signal time series')
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.plot(Time, signal)
            plt.show()
        
        # Returns channel information if you want
        if return_info:
            return (signal, fs, seconds, frames)


# In[48]:


# window and overlap time are in seconds
# pretty sure this is the good one :) 
def make_spectrogram(signal, fs, window_time = 0.04, overlap_time = 0.02, min_x = 0, 
                     max_x = 8, min_y = 0, max_y = 8000, dyn_range = 120, make_plot = True):
    
    x = np.array(signal)
    window_length = int(window_time * fs)
    overlap_length = int(overlap_time * fs)
    f, t, intensity = spectrogram(x, fs, nperseg = window_length, noverlap = overlap_length)
    np.place(intensity, intensity == 0, [1e-300])
    
    logIntensity = np.log10(intensity)
    
    if make_plot:
        p_ref = 2e-5
        dB_max = 10 * (logIntensity.max() - 2*np.log10(p_ref))
        vmin = p_ref * 10 ** ((dB_max - dyn_range) / 10)

        # other arguments to the figure
        extent = (t.min(), t.max(), f.min(), f.max())
        plt.ylim(min_y,max_y)
        plt.xlim(min_x,max_x)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.imshow(logIntensity, origin='lower', aspect='auto', cmap='hot_r',
                   extent=extent, vmin=vmin, vmax=None)

    return f, t, logIntensity


# In[49]:


def voss(nrows, ncols=16):
    """Generates pink noise using the Voss-McCartney algorithm.
    
    nrows: number of values to generate
    rcols: number of random sources to add
    
    returns: NumPy array
    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values


# In[50]:


# plot a time series for your audio signal array
def plot_waveform(signal, fs, t_i, t_f):
    
    frames = fs*(t_f - t_i)
    Time = np.linspace(t_i, t_f, num = frames)

    plt.title('Signal time series')
    plt.plot(Time, signal[t_i*fs : t_f*fs])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


# In[51]:


def plot_FT(signal, freq_bound, return_FT = False, convolve = False, wind = 0):
    ft_sig = np.abs(np.fft.fft(signal))
    ft_until = ft_sig[0:freq_bound]
    
    if not convolve:
        plt.plot(ft_until)
        if return_FT:
            return ft_until
        
    else:
        the_wind = np.hanning(wind)
        smooth = np.convolve(ft_until, the_wind)
        plt.plot(smooth)
        if return_FT:
            return smooth


# In[52]:


def spectral_sub (file =  "", signal = [], fs = 0, new_file_name = "", noise_int = (0,1), 
              frame_time = 0.04, overlap = 0.5, p = 1, alpha = 1, beta = 0.15, window = sp.hanning, 
              compare_specs = True, return_spec = False, spec_window_time = 0.04, 
              spec_overlap_time = 0.02, compare_waveforms = False, return_signal = False):
              
    
    if file:
        if signal != [] or fs != 0 :
            raise ValueError("Must pass either only file or only signal. \nIf passing signal, make sure to pass fs as well.")
        else:
            fs, signal = wf.read(file)

            wav_file = wave.open(file,'r')

            # Grab audio data from first audio channel if its a stereo signal
            if wav_file.getnchannels() == 2:
                signal = signal[:,0]

            wav_file.close()
    else:
        if signal == [] or fs == 0:
            raise ValueError("Must pass either only file or only signal. \nIf passing signal, make sure to pass fs as well.")
    
    frames = len(signal)

    # Length of audio file
    seconds = frames/fs

    # Creates interval spanning channel with time instance at each frame 
    Time = np.linspace(0, seconds, num = frames)
    
    my_noise = signal[int(noise_int[0]*fs) : int(noise_int[1]*fs)]
    my_signal = signal

    # Frame length in seconds -> number of frames
    frame_length = int(frame_time*fs)
    noise_len = len(my_noise)

    # Gets padding length and pads noise
    rem_frames = frame_length - noise_len%frame_length
    new_noise = np.pad(my_noise,(0, int(rem_frames)),"constant")
    new_noise_len = len(new_noise)

    # Number of noise frames based on frame length 
    noise_frames = int(new_noise_len/frame_length)
    total_FFT = [0 for i in range(frame_length)]

    # Gets average frequency distribution of the noise
    for i in range(noise_frames):
        sample = new_noise[i*frame_length : (i+1)*frame_length]
        sample_FFT = sp.fft(sample)#*window(len(sample)))
        total_FFT += np.abs(sample_FFT)**p

    avrg_noise = total_FFT/(noise_len/frame_length)

    # Overlap percentage of frame between 50% (halfway overlap) and 100% (no overlap)
    overlap_frames = int((1-overlap)*frame_length)
    complement_overlap_frames = frame_length - overlap_frames

    # Gets padding length and pads signal
    signal_len = len(my_signal)
    rem_signal_frames = overlap_frames - (signal_len % overlap_frames) 
    new_signal = np.pad(my_signal, (0, int(rem_signal_frames)), "constant")
    new_signal_len = len(new_signal)

    # Number of signal bins based on frame length and overlap
    signal_bins = int(1 + (new_signal_len - frame_length)/overlap_frames)

    the_clean_signal = []
    # FFT's frame, subtracts average noise from it, IFFT's frame
    # Moves by unit of overlap length
    
    for i in range(signal_bins):
        sample = new_signal[i*overlap_frames : i*overlap_frames + frame_length]
        sample_FFT = sp.fft(sample*window(len(sample)))

        # Spectral over/undersubtraction and noise spectral floor parameter
        clean_sample_FFT = np.abs(sample_FFT)**p - alpha*avrg_noise
        clean_sample_FFT[clean_sample_FFT < 0] = beta*avrg_noise[clean_sample_FFT < 0]

        sample_phase = np.angle(sample_FFT)
        clean_sample_phase = np.exp(1j*sample_phase)
        pth_root_sample = clean_sample_FFT**(1/p)
        clean_sample = sp.real(sp.ifft(clean_sample_phase*pth_root_sample))
        the_clean_signal.append(clean_sample)

    final_clean_signal = []

    if overlap != 1:
        # Stitches signal back together in time domain by averaging overlaps
        final_clean_signal.append(the_clean_signal[0][0:complement_overlap_frames])
        for i in range(len(the_clean_signal)-1):
            # Average
            a = the_clean_signal[i][complement_overlap_frames:]
            b = the_clean_signal[i+1][0 : overlap_frames]
            c = np.mean([a,b], axis = 0)
            final_clean_signal.append(c)

            # Rest of current array
            final_clean_signal.append(the_clean_signal[i+1][overlap_frames : complement_overlap_frames])
        
        # Last snippit of final frame before zero pad begins
        final_clean_signal.append(the_clean_signal[i+1][complement_overlap_frames:-rem_signal_frames])
        #final_clean_signal.append(the_clean_signal[i+1][complement_overlap_frames:])

        final_clean_signal = [item for sublist in final_clean_signal for item in sublist]

    else: 
        final_clean_signal = [item for sublist in the_clean_signal for item in sublist]

    # Compares waveforms, if you want
    if compare_waveforms:
        plt.title('Original signal time series')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.plot(Time, my_signal)
        plt.show()

        plt.title('Clean signal time series')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.plot(Time, final_clean_signal)
        plt.show()
        
    # Comparing spectrograms, or just returning the clean spectrogram, if you want
    if compare_specs == True:
        F_old, T_old, I_old = make_spectrogram(signal, fs, spec_window_time, spec_overlap_time)
        plt.show()
        F, T, I = make_spectrogram(final_clean_signal, fs, spec_window_time, spec_overlap_time)
        if return_spec == True:
            return I
    else:
        if return_spec == True:
            F, T, I = make_spectrogram(final_clean_signal, fs, spec_window_time, spec_overlap_time, make_plot=False)
            return I
        
    # Write new audio to file, if you want 
    if new_file_name:
        scaled = np.int16(final_clean_signal/np.max(np.abs(final_clean_signal)) * 32767)
        wf.write(new_file_name, fs, scaled)

    # Return clean signal, if you want
    if return_signal:
        return final_clean_signal


# This section includes old versions of methods.

# def create_CAR_original(subj, block, bad_elecs, root_path):
#     if subj.startswith("NY"): #checks for NYU subject IDS
#         if create_dir==True:
#             if root_path.endswith(subj):
#                 data_path = op.join(root_path, "data", block, "gdat.h5")
#             elif root_path.endswith("subjsPython"):
# 
#                 data_path = op.join(root_path, subj, "data", block, "gdat.h5")
#             else:
#                 data_path = op.join(root_path, "subjsPython", subj, "data", block, "gdat.h5")
#         else:
#             data_path = op.join(root_path, "data", block, "gdat.h5")
#     else:
#         raise ValueError("Subject number must start with NY")
#             
#     data = load_h5(data_path, "gdat")
#     good_data = [data[i] for i in range(len(data)) if i not in bad_elecs]
#     good_data_zero_meaned = [good_data[i] - np.mean(good_data[i]) for i in range(len(good_data))]
#     reference = np.mean(good_data_zero_meaned, axis = 0)
#     reference.resize((1,np.shape(reference)[0]))
#     new_data = np.subtract(good_data_zero_meaned, reference)
#     
#     #saving
#     if root_path.endswith(subj):
#             
#         save_h5(op.join(root_path, "data", block,'car.h5'), "car", reference)
#         print('saving CAR')
#         save_h5(op.join(root_path, "data", block, 'car_data.h5'), "car_data", new_data)
#         print('saving Car Data')
#     elif root_path.endswith("subjsPython"):
#         save_h5(op.join(root_path, subj, "data", block,'car.h5'), "car", reference)
#         print('saving CAR')
#         save_h5(op.join(root_path, subj, "data", block, 'car_data.h5'), "car_data", new_data)
#         print('saving Car Data')
#             
#     else:
#         save_h5(op.join(root_path, "subjsPython", subj, "data", block,'car.h5'), "car", reference)
#         print('saving CAR')
#         save_h5(op.join(root_path, "subjsPython", subj, "data", block, 'car_data.h5'), "car_data", new_data)
#         print('saving Car Data')
#     
#     
#     return new_data, reference

# def extract_blocks_original(trigger_name, subj, srate, blockMin=90000, eventMin=256,gap=15000, trigger_len=2600):
#     
#     data,times = raw.copy().pick_channels([trigger_name])[:,:]
#     trig_str= ['picN_trigger','visRead_trigger','audRep_trigger','audN_trigger','senComp_trigger']
#     task=['picN', 'visRead', 'audRep','audN', 'senComp']
# 
#     #loading triggers from file
#     picN_trigger=load_h5(op.join(home_path,'picN_trigger.h5'),'picN_trigger')
#     visRead_trigger=load_h5(op.join(home_path,'visRead_trigger.h5'),'visRead_trigger')
#     audRep_trigger=load_h5(op.join(home_path,'audRep_trigger.h5'),'audRep_trigger')
#     audN_trigger=load_h5(op.join(home_path,'audN_trigger.h5'),'audN_trigger')
#     senComp_trigger=load_h5(op.join(home_path,'senComp_trigger.h5'),'senComp_trigger')
#     
#     triggers= [picN_trigger,visRead_trigger,audRep_trigger,audN_trigger,senComp_trigger]
#     
#     picN_lag = np.argmax(correlate(data[0], picN_trigger))+trigger_len
#     visRead_lag= np.argmax(correlate(data[0], visRead_trigger))+trigger_len
#     audRep_lag=np.argmax(correlate(data[0], audRep_trigger))+trigger_len
#     audN_lag=np.argmax(correlate(data[0], audN_trigger))+trigger_len
#     senComp_lag=np.argmax(correlate(data[0], senComp_trigger))+trigger_len
#     lags= [picN_lag,visRead_lag, audRep_lag, audN_lag,senComp_lag]
#     for i,t in enumerate(triggers):
#         if max(correlate(data[0], t))<20: 
#             lags.remove(lags[i])
#     lags= [picN_lag,visRead_lag, audRep_lag, audN_lag,senComp_lag]
#     enum_lags = list(enumerate(lags))
#     sorted_lags=sorted(enum_lags, key=lambda x: x[1])
#     lag_tags = ['picN_lag','visRead_lag', 'audRep_lag', 'audN_lag','senComp_lag']
#     sorted_tags=[lag_tags[i] for i,j in sorted_lags]
#     task=[task[i] for i,j in sorted_lags]
#     
#     print('Found {0} blocks for {1}\n'.format(len(lags), subj))
#     
#     block_times = []
#     
#     for t,v in enumerate(lags):
#         try:
#             block_times.append([int(v),int(lags[t+1]-gap)])
#         except:
#             if int(v)+gap <= len(data):
#                 block_times.append([int(v),int(v)+gap])
#                 
#             else:
#                 block_times.append([int(v),int(times[-1]*srate)])
#                 
#     
#     print('Block order followed by index times are as follows: {0} \n {1}'.format(task[:], block_times[:]))
#      
#     return block_times

# def another_extract_blocks_old(trigger_name, subj, srate=512, blockMin=90000, eventMin=256,gap=15000, trigger_len=2600):
#     data,times = raw.copy().pick_channels([trigger_name])[:,:]
#     data = data-np.mean(data)#mean center
#     data= data/abs(max(data.T))#normalize
#     task=['picN', 'visRead', 'audRep','audN', 'senComp']
#     i=0
#     triggers=0
#     blocks=[]
#     while i< len(times):
#         j=0
#         if data.T[i]>.7:
#             hit=times[i]*srate # marks index-time for trigger 
#             i=i+70       #advances more than length of trigger spike
#             triggers+=1
#             while j<50:  #searches for next spike of block-level trigger
#                 j=j+1
#                 if data.T[i+j]>.7: # if found, mark the hit time
#                     blocks.append(hit+trigger_len)
#                     i=i+blockMin  #advance a little below the minimum block time
#                     j=51          # exit j-loop
#                     #### need to account for finding task triggers
#             i=i+50
#         i=i+1
#     blocks=np.asarray(blocks)
#     blocks=blocks.astype(int)
#     blockSums= [int(sum(abs(data[0][(blocks[t]-trigger_len):(blocks[t])]))) for t,v in enumerate(blocks)]
# 
#     taskorder=[]
#     for i in blockSums:
#         if i > 139:
#             taskorder.append('picN')
#         elif i >100:
#             taskorder.append('visRead')
#         elif i >40:
#             taskorder.append('audRep')
#         elif i >30:
#             taskorder.append('audN')
#         elif i  >20:
#             taskorder.append('senComp')       
#                 
#     print('blockS', blockSums)
#     print('Found {0} blocks for {1}\n'.format(len(blocks), subj))
#     block_times=[]
#     for t,v in enumerate(blocks):
#         try:
#             block_times.append([int(v),int(blocks[t+1]-gap)])
#         except:
#             if hit+gap <= len(data):
#                 block_times.append([int(v),int(hit+gap)])
#                 
#             else:
#                 block_times.append([int(v),int(times[-1]*srate)])
#                 
#         
#     print('Block order followed by index times are as follows: {0} \n {1}'.format(taskorder[:], block_times[:]))    
#     return block_times

# # window and overlap time are in seconds
# # plots in log scale, doesn't calculate intensity in log scale
# def make_spectrogram_old(signal, fs, window_time = 0.04, overlap_time = 0.02, min_x = 0, 
#                      max_x = 30, min_y = 0, max_y = 8000, dyn_range = 120, make_plot = True):
#     
#     x = np.array(signal)
#     window_length = int(window_time * fs)
#     overlap_length = int(overlap_time * fs)
#     f, t, intensity = spectrogram(x, fs, nperseg = window_length, noverlap = overlap_length)
#     #np.place(intensity, intensity == 0, [0.000001])
#     #intensity[intensity == 0] = 0.00001
#     #logIntensity = np.log10(intensity)
#     
#     if make_plot:
#         p_ref = 2e-5
#         dB_max = 10 * np.log10(intensity.max() / (p_ref ** 2))
#         vmin = p_ref * 10 ** ((dB_max - dyn_range) / 10)
# 
#         # other arguments to the figure
#         extent = (t.min(), t.max(), f.min(), f.max())
#         plt.ylim(min_y,max_y)
#         plt.xlim(min_x,max_x)
#         plt.ylabel('Frequency [Hz]')
#         plt.xlabel('Time [sec]')
#         plt.imshow(intensity, origin='lower', aspect='auto', cmap='hot_r',
#                   norm=LogNorm(), extent=extent, vmin=vmin, vmax=None)
# 
#     return f, t, intensity

# # Time delays the data, returns time delayed data matrix and response vector
# def time_delay_data_old(data_matrix, response_vector, remaining_frames, wind_len):
#     
#     new_X = []
#     new_y = []
#     
#     # Properly construct the X matrix (concat trials together (need to implement))
#     for ind, spec in enumerate(data_matrix):
#         x = []
#         for k in range(remaining_frames):
#             x.append(spec[: , k:k+wind_len])
#         x = np.concatenate(x, axis = 0)
#         new_X.append(x)
#         new_y.append(y[ind][0:wind_len])
#         
#     # These are the y and X data we want to work with
#     new_X = np.asarray(new_X)
#     new_y = np.asarray(new_y)    
#     
#     return new_X, new_y

# # plot the Fourier Transform of your signal (given it's in the frequency domain)
# def plot_FT_old(FT_signal, fs, min_x, max_x, min_y, max_y):
#     
#     # period is time between frames
#     period = 1/fs
#     N = len(FT_signal)
#     
#     # Get frequencies and sort them
#     freqs = np.fft.fftfreq(N, period)
#     idx = np.argsort(freqs)
#     
#     # Power spectrum
#     power_spec = np.abs(FT_signal)
#     
#     # Plotting
#     plt.title('Signal Fourier Transform')
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Intensity")
#     plt.xlim(min_x,max_x)
#     plt.ylim(min_y,max_y)
#     plt.plot(freqs[idx], power_spec[idx])
