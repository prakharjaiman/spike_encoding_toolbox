# Author: Nikhil Garg
# Organization: 3IT & NECOTIS,
# Université de Sherbrooke, Canada

import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import gc
import matplotlib.pyplot as plt
import time
from utilis import *
from args import args as my_args
from encode import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import scikitplot as skplt
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_selection import RFE,SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics as metrics
# import autosklearn.classification
from evaluate_encoder import  *

def run_random_experiment(seed):
    args = my_args()
    
    np.random.seed(seed)
    seed=np.random.randint(0, 0xFFFFFFF),
    args.encode_thr_up=np.random.choice([0, 150])
    args.encode_thr_dn=np.exp(-1 / np.random.uniform(5, 200))
    args.tstep=np.random.hoice([0.0, 0.05, 0.1, 0.3, 0.5, 1.0])
    args.interpfact=np.random.choice([0.0, 0.05, 0.1, 0.3, 0.5, 1.0])
    args.refractory=np.random.choice([0.0, 0.05, 0.1, 0.3, 0.5, 1.0]) 
    input_args=[args.encode_thr_up,args.encode_thr_dn,args.tstep,args.interpfact,args.refractory]
    input_args_string=map(str,input_args)
    output_file = f"ecog_{seed}.txt"
    if os.path.exists(output_file):
        exit(0)

    svm_score_input,rf_score_input,avg_spike_rate, svm_score_baseline, svm_score_comb, rf_score_comb = evaluate_encoder(args)
    with open(output_file, "w") as f:
        results=[svm_score_input,rf_score_input,avg_spike_rate, svm_score_baseline, svm_score_comb, rf_score_comb]
        f.write(str(input_args_string))
        f.write(str(results))

if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    args = my_args()
    print(args.__dict__)
    logging.basicConfig(level=logging.DEBUG)
    # Fix the seed of all random number generator
    seed = 500
    random.seed(seed)
    np.random.seed(seed)
    svm_score_input,rf_score_input,avg_spike_rate, svm_score_baseline, svm_score_comb, rf_score_comb= evaluate_encoder(args)
    print('Average spike rate :')
    print(avg_spike_rate)
    print('Accuraccy input: ' + str(svm_score_input))
    logger.info('All done.')