#!/usr/bin/env python

## ex: results.py <test.csv> <dot_product_sums.npy> <label> <threshold> // the second file is a np.array

import sys
from datetime import datetime as dt
from time import time
import numpy as np
import pandas as pd
from warnings import filterwarnings

filterwarnings("ignore")


def main():
    csv_in = sys.argv[1]
    datafile = sys.argv[2]
    label = sys.argv[3]
    threshold = float(sys.argv[4])
    data = np.load(datafile)
    df = pd.read_csv(csv_in)
    ## select threshold to find matches
    predicts = find_matches(data, df, threshold)
    
    ## calculate accuracy, precision, recall, f-measure
    ## need TP=#correct predict, TN=#correct reject,
    ## FP=#wrong predict (type 1 err), FN=#wrong reject (type 2 err)
    num = len(df.loc[df.account_type == label])
    tot = len(df)

    TP = len(np.where(predicts == label)[0])
    FP = len(predicts) - TP
    FN = num-TP
    TN = tot - num - FP

    accu = accuracy(TP, TN, tot)
    prec = precision(TP, FP)
    reca = recall(TP, FN)
    f_me = f_measure(TP, FN, FP)
    print("accuracy: %.3f\nprecision: %.3f\nrecall: %.3f\nf-measure: %.3f\n"%(accu, prec, reca, f_me))
    

def accuracy(TP, TN, tot):
    return ( TP + TN ) / tot


def precision(TP, FP):
    return TP / ( TP + FP )


def recall(TP, FN):
    return TP / ( TP + FN )
    

def f_measure(TP, FN, FP):
    return 2 / ( 1 / precision(TP, FP) + 1 / recall(TP, FN) )
    

def find_matches(data, df ,thres):
    return np.array([df.iloc[i].account_type for i in range(len(data)) if data[i] > thres])
 

if __name__=="__main__":
    t1 = time()
    main()
    print(time()-t1)
