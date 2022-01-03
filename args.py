# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Nikhil
"""

import argparse


def args():
    parser = argparse.ArgumentParser(
        description="Train a reservoir based SNN on biosignals"
    )

    # Defining the model
    parser.add_argument(
        "--dataset", default="bci3", type=str, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--encode_thr_up",
        default=0.5,
        type=float,
        help="Threshold UP for spike encoding" "e.g. 0.25, 0.50 etc.",
    )
    parser.add_argument(
        "--encode_thr_dn",
        default=0.5,
        type=float,
        help="Threshold UP for spike encoding" "e.g. 0.25, 0.50 etc.",
    )
    parser.add_argument(
        "--encode_refractory",
        default=1,
        type=float,
        help="Refractory period for spike encoding" "e.g. 0.25, 0.50 etc.",
    )
    parser.add_argument(
        "--encode_interpfact",
        default=1,
        type=float,
        help="Interpolation factor in ms for spike encoding" "e.g. 1, 2, 3, etc.",
    )

    parser.add_argument(
        "--encoded_data_file_prefix",
        default='bci3_encoded',
        type=str,
        help="",
    )
    

    
    parser.add_argument(
        "--tstep",
        default=200,
        type=float,
        help="Readout layer step time in ms" "e.g. 200, 300, etc etc.",
    )


    parser.add_argument(
        "--tstart",
        default=0,
        type=float,
        help="Time point from which the simulated sub-segment(of length tstep) is used as a feature for readout layer" ">0 (in ms).",
    )

    parser.add_argument(
        "--tlast",
        default=3000,
        type=float,
        help="Time point till which the simulated sub-segment(of length tstep) is used as a feature for readout layer" "e.g. <1800> (in ms).",
    )
    parser.add_argument(
        "--duration",
        default=3000,
        type=float,
        help="Time point till which the simulation has to be run",
    )

    parser.add_argument(
        "--seed",
        default=50,
        type=float,
        help="Seed for random number generation",
    )


    parser.add_argument('--experiment_name', default='standalone', type=str,
                        help='Name for identifying the experiment'
                               'e.g. plot ')


    parser.add_argument('--fold', default=3, type=float,
                        help='Fold for train/test'
                             'e.g. 1, 2, 3 ')


    parser.add_argument('--log_file_path', default=None, 
                        help='Path for log file')


    my_args = parser.parse_args()

    return my_args
