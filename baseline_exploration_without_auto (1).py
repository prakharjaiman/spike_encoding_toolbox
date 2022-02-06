import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os

#from evaluate_reservoir import *
from utilis import *
from args import args as my_args
from evaluate_encoder import  *
from itertools import product
import time

if __name__ == '__main__':

  args = my_args()
  print(args.__dict__)
	# Fix the seed of all random number generator
  seed = 50
  random.seed(seed)
  np.random.seed(seed)
  df = pd.DataFrame({	"dataset":[],"encode_thr_up":[],"encode_thr_dn":[],"tstep":[],"encode_refractory":[],"encode_interpfact":[],"firing_rate":[],"svm_score":[],"rf_score":[],"svm_score_baseline":[],"svm_score_comb":[],"rf_score_comb":[],"gen_accuracy":[],"generation":[],"selected_features":[],"genetic_final_accuracy":[],"n_selected_features":[]})

  parameters = dict(
		dataset = ["bci3"]
		,encode_thr_up = [1.1]
    	,encode_thr_dn = [1.1]
		,tstep=[3000]
		,interpfact = [1]
		,refractory = [1]
    )
  param_values = [v for v in parameters.values()] 
  for args.dataset,args.encode_thr_up,args.encode_thr_dn, args.tstep, args.encode_interpfact,args.encode_refractory in product(*param_values):
    args.experiment_name = str(args.dataset)+str(args.encode_thr_up)+str(args.encode_thr_dn)+str(args.encode_interpfact)+str(args.encode_refractory)
    svm_score, rf_score, firing_rate, svm_score_baseline, svm_score_comb, rf_score_comb,acc,sel,gen,nfeat = evaluate_encoder(args)
    for n in range(args.gen+1):
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
              "gen_accuracy":gen[str(n)],
              "generation":n,
              "selected_features":sel,
              "genetic_final_accuracy":acc,
              "n_selected_features":nfeat
                      },ignore_index=True)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = 'accuracy_log'+str(timestr)+'.csv'
    pwd = os.getcwd()
    log_dir = pwd+'/log_dir/'
    df.to_csv(log_dir+log_file_name, index=False)

  df.to_csv(log_file_name, index=False)
	# logger.info('All done.')
