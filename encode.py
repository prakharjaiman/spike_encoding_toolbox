#!/usr/bin/env python
# coding: utf-8

# spike conversion algorithm.
# Spike time array contains values of spike times in ms.
# Saved arrays :

# X: Array of the EMG/EEG/ECoG Digital time series data with length = 200
# Y: Array of the labels of theing data with length = 200

# spike_times_up: Spike time arrays with upward polarity in ms for X. length = 200
# spike_times_dn: Spike time arrays with downward polarity in ms for X. length = 200

# Author : Nikhil Garg, 3IT Sherbrooke ; nikhilgarg.bits@gmail.com
# Created : 15 July 2020
# Last edited : 3rd January 2022

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.signal import butter, lfilter, welch, square  # for signal filtering
from utilis import *
from args import args as my_args

def encode(args):
    # general stuff
    # sampling frequency of MYO
    
    VERBOSE = True
    pwd = os. getcwd()

    if args.dataset == "bci3":
        data_dir = pwd + "\\dataset\\bci_3.npz"
        fs = 1000
        nb_channels = 64
        
    else:
        print("Invalid dataset")
        fs = 1000

    #Add data here
    X_Train = []
    Y_Train = []
    X_Test = []
    Y_Test = []

    data = np.load(data_dir)
    X_Train = data['X']
    Y_Train = data['events']

    X_Test = data['X']
    Y_Test = data['events']



    X_Train = np.array(X_Train)

    X_Train = np.moveaxis(X_Train, 2, 1)
    Y_Train = np.array(Y_Train)

    X_Test = np.array(X_Test)
    X_Test = np.moveaxis(X_Test, 2, 1)
    Y_Test = np.array(Y_Test)

    # X_uniform is a time series data array with length of 400. The initial segments are about 397, 493 etc which
    # makes it incompatible in some cases where uniform input is desired.

    nb_trials = X_Train.shape[0]

        
    # print(len(X))
    print("Number of training samples in dataset:")
    print(len(X_Train))
    print(len(Y_Train))
    # print("Class labels:")
    # print(list(set(Y_Train)))

    # Take session 0,1 as and session 2 as test.


    interpfact = args.encode_interpfact
    refractory_period = args.encode_refractory  # in ms
    th_up = args.encode_thr_up
    th_dn = args.encode_thr_dn


    # Generate the  data
    X=X_Train
    Y=Y_Train
    spike_times_train_up = []
    spike_times_train_dn = []
    for i in range(len(X)):
        spk_up, spk_dn = gen_spike_time(
            time_series_data=X[i],
            interpfact=interpfact,
            fs=fs,
            th_up=th_up,
            th_dn=th_dn,
            refractory_period=refractory_period,
        )
        spike_times_train_up.append(spk_up)
        spike_times_train_dn.append(spk_dn)
    

    rate_up = gen_spike_rate(spike_times_train_up)
    rate_dn = gen_spike_rate(spike_times_train_dn)
    avg_spike_rate = (rate_up+rate_dn)/2
    print("Average spiking rate")
    print(avg_spike_rate)

        # Generate the  data
    X=X_Test
    Y=Y_Test
    spike_times_test_up = []
    spike_times_test_dn = []
    for i in range(len(X)):
        spk_up, spk_dn = gen_spike_time(
            time_series_data=X[i],
            interpfact=interpfact,
            fs=fs,
            th_up=th_up,
            th_dn=th_dn,
            refractory_period=refractory_period,
        )
        spike_times_test_up.append(spk_up)
        spike_times_test_dn.append(spk_dn)
    



    nb_trials = X_Train.shape[0]

        
    # print(len(X))
    print("Number of test samples in dataset:")
    print(len(X_Test))
    print(len(Y_Test))
    # print("Class labels:")
    # print(list(set(Y_Test)))


    spike_times_train_up = np.array(spike_times_train_up)
    spike_times_test_up = np.array(spike_times_test_up)
    spike_times_train_dn = np.array(spike_times_train_dn)
    spike_times_test_dn = np.array(spike_times_test_dn)


    file_path = "dataset/"
    file_name = args.encoded_data_file_prefix + str(args.dataset) + str(args.encode_thr_up) + str(
        args.encode_thr_dn) + str(args.encode_refractory) + str(args.encode_interpfact) + ".npz"

    np.savez_compressed(
        file_path + file_name,
        # X_Train=X_Train,
        Y_Train=Y_Train,
        # X_Test=X_Test,
        Y_Test=Y_Test,
        spike_times_train_up=spike_times_train_up,
        spike_times_train_dn=spike_times_train_dn,
        spike_times_test_up=spike_times_test_up,
        spike_times_test_dn=spike_times_test_dn,
    )
    return spike_times_train_up, spike_times_train_dn, spike_times_test_up, spike_times_test_dn, X_Train,X_Test, Y_Train,Y_Test,avg_spike_rate

if __name__ == '__main__':
    args = my_args()
    print(args.__dict__)
    # Fix the seed of all random number generator
    encode(args)
