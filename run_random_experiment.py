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

def run_random_experiment(args):
    args = my_args()
    seed = int(args.seed)
    
    np.random.seed(seed)
    # seed=np.random.randint(0, 0xFFFFFFF),
    args.encode_thr_up=np.random.uniform(0.5, 2)
    args.encode_thr_dn=np.random.uniform(0.5, 2)
    args.tstep=np.random.choice([200,300,500,600,1000,1500,3000])
    args.interpfact=np.random.choice([1,2,3])
    args.refractory=np.random.choice([0,1,2]) 
    input_args=[args.encode_thr_up,args.encode_thr_dn,args.tstep,args.interpfact,args.refractory]
    input_args_string=map(str,input_args)
    output_file = f"ecog_{seed}.csv"
    # # if os.path.exists(output_file):
    # #     exit(0)

    svm_score_input,rf_score_input,avg_spike_rate, svm_score_baseline, svm_score_comb, rf_score_comb = evaluate_encoder(args)
    results=[svm_score_input,rf_score_input,avg_spike_rate, svm_score_baseline, svm_score_comb, rf_score_comb]
    df = pd.DataFrame({	"dataset":[],"encode_thr_up":[],"encode_thr_dn":[],"tstep":[],"encode_refractory":[],"encode_interpfact":[],"firing_rate":[],"svm_score":[],"rf_score":[],"svm_score_baseline":[],"svm_score_comb":[],"rf_score_comb":[], "auto_score":[]})
    df = df.append({ "dataset":args.dataset,
						 "fold":args.fold,
						 "encode_thr_up":args.encode_thr_up,
						 "encode_thr_dn":args.encode_thr_dn,
						 "tstep": args.tstep,
						 "encode_refractory": args.encode_refractory,
						 "encode_interpfact": args.encode_interpfact,			 
		                 "firing_rate":firing_rate,
		                 "svm_score":svm_score,
                     	 "rf_score":rf_score,
		                 "svm_score_baseline":svm_score_baseline,
						 "svm_score_comb":svm_score_comb,
						 "rf_score_comb":rf_score_comb,
						#  "auto_score":auto_score
		                 },ignore_index=True)
    df.to_csv(output_file, index=False)
        # f.write('haha')
        

if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    args = my_args()
    seed=int(args.seed)
    print(args.__dict__)
    logging.basicConfig(level=logging.DEBUG)
    # Fix the seed of all random number generator
    # seed = 500
    random.seed(seed)
    np.random.seed(seed)
    run_random_experiment(args)
    # print('Average spike rate :')
    # print(avg_spike_rate)
    # print('Accuraccy input: ' + str(svm_score_input))
    logger.info('All done.')