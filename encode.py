#!/usr/bin/env python
# coding: utf-8

# spike conversion algorithm.
# Spike time array contains values of spike times in ms.
# Saved arrays :

# X: Array of the EMG Digital time series data with length = 300
# Y: Array of the labels of theing data with length = 300

# spike_times_up: Spike time arrays with upward polarity in ms for X. length = 300
# spike_times_dn: Spike time arrays with downward polarity in ms for X. length = 300

# Author : Nikhil Garg, 3IT Sherbrooke ; nikhilgarg.bits@gmail.com
# Created : 15 July 2020
# Last edited : 12th September 2020

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
    fs = 200
    VERBOSE = True
    # pwd = os. getcwd()

    if args.dataset == "roshambo":
        data_dir = "dataset/Gesture_5_class/"
    elif args.dataset == "5_class":
        data_dir = "dataset/Gesture_5_class/"
        
    else:
        print("Invalid dataset")

    #Add data here
    X = []
    Y = []
    X = np.array(X)
    Y = np.array(Y)

    # X_uniform is a time series data array with length of 400. The initial segments are about 397, 493 etc which
    # makes it incompatible in some cases where uniform input is desired.

    nb_trials = X.shape[0]
    len_trial = fs * 2  # 2 seconds of trial, sampling rate is 200
    nb_channels = 8
    X_uniform = np.ones((nb_trials, len_trial, nb_channels))
    for i in range(len(X)):
        trial_length = X[i].shape[0]
        if trial_length > len_trial:
            X_uniform[i] = X[i][0:len_trial]
        elif trial_length < len_trial:
            short = len_trial - trial_length
            pad = np.zeros((short, nb_channels))
            X_uniform[i] = np.append(X[i], pad, axis=0)
        else:
            X_uniform[i] = X[i]
    # print(len(X))
    print("Number of samples in dataset:")
    print(len(X_uniform))
    print(len(Y))
    print("Class labels:")
    print(list(set(Y)))

    # Take session 0,1 as and session 2 as test.


    interpfact = args.encode_interpfact
    refractory_period = args.encode_refractory  # in ms
    th_up = args.encode_thr_up
    th_dn = args.encode_thr_dn
    n_ch = 8
    fs = 200

    # Generate the  data
    spike_times_up = []
    spike_times_dn = []
    for i in range(len(X)):
        spk_up, spk_dn = gen_spike_time(
            time_series_data=X[i],
            interpfact=interpfact,
            fs=fs,
            th_up=th_up,
            th_dn=th_dn,
            refractory_period=refractory_period,
        )
        spike_times_up.append(spk_up)
        spike_times_dn.append(spk_dn)
    

    rate_up = gen_spike_rate(spike_times_up)
    rate_dn = gen_spike_rate(spike_times_up)
    avg_spike_rate = (rate_up+rate_dn)/2
    print("Average spiking rate")
    print(avg_spike_rate)


    _t = np.arange(
        0, 2000, 5
    )  # Time array of 2000ms for the 200 samples per second. For ploting purpose.
    _t_spike = np.arange(0, 2000, 1)  # Time array for defining the X axis of graph.

    # Plot a up segment
    plt.eventplot(spike_times_up[1], color=[0, 0, 1], linewidth=0.5)
    plt.xlabel("Time(ms)")
    plt.ylabel("Channel")
    plt.title("Spike raster plot for up channel")

    # Plot a dn segment
    plt.eventplot(spike_times_dn[1], color=[1, 0, 0], linewidth=0.5)
    plt.xlabel("Time(ms)")
    plt.ylabel("Channel")
    plt.title("Spike raster plot for down channel")

    channels = np.linspace(0, nb_channels-1, num=nb_channels)
    
    plt.plot(_t, X[1], linewidth=0.5)
    plt.legend(channels)
    plt.title("Raw Data")
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude")
    pwd = os.getcwd()
    fig_dir = pwd + '/plots/'
    fig_name = 'encoded-data' + str(args.dataset) + str(args.encode_thr_up) + str(args.encode_thr_dn) + str(
        args.encode_refractory) + str(args.encode_interpfact)+str(args.fold) + ".svg"

    plt.savefig(fig_dir+fig_name)
    plt.clf()


    spike_times_up = np.array(spike_times_up)
    spike_times_up = np.array(spike_times_up)


    file_path = "dataset/"
    file_name = args.encoded_data_file_prefix + str(args.dataset) + str(args.encode_thr_up) + str(
        args.encode_thr_dn) + str(args.encode_refractory) + str(args.encode_interpfact)+ str(args.fold) + ".npz"

    np.savez_compressed(
        file_path + file_name,
        X_Train=X_Train,
        Y_Train=Y_Train,
        X_Test=X_Test,
        Y_Test=Y_Test,
        spike_times_up=spike_times_up,
        spike_times_dn=spike_times_dn,
    )
    return spike_times_train_up, spike_times_train_dn, spike_times_test_up, spike_times_test_dn, X_Train,X_Test, Y_Train,Y_Test,avg_spike_rate

if __name__ == '__main__':
    args = my_args()
    print(args.__dict__)
    # Fix the seed of all random number generator
    encode(args)
