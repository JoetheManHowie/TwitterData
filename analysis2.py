#!/usr/bin/env python

## ex: analysis2.py <train.csv> <test.csv>

from time import time
from datetime import datetime as dt
from MyVecs import MyTfidfVectorizer
import sys
import pandas as pd
import numpy as np

def main():
    train = sys.argv[1]
    test = sys.argv[2]
    savefile = train.split(sep='.')[0]
    #### import file ####
    print("---- reading data files ----", dt.now())
    df_train = pd.read_csv(train)
    df_test = pd.read_csv(test)
    #### make training vectorizer ####
    print("---- constructing training vector ----", dt.now())
    vec_train = MyTfidfVectorizer(ngram_range=(1,2),
                                  stop_words='english',
                                  vocabulary=bow(df_train.content))
    tfidf_train = vec_train.fit_transform(df_train.content)
    #### make test vectorizer ####
    print("---- make test vectorizer ----", dt.now())
    vec_test = MyTfidfVectorizer(ngram_range=(1,2),
                                  stop_words='english',
                                  vocabulary=vec_train.get_feature_names())
    tfidf_test = vec_test.fit_transform(df_test.content)
    #### calculate dot products ####
    print("---- calculate dot products ----", dt.now())
    dots = tfidf_test.dot(tfidf_train.T)
    one = np.ones((len(df_train), 1))
    print("dots", np.shape(dots))
    print("one", np.shape(one))
    sums = dots.dot(one)
    print(np.shape(sums))
    np.save(savefile+".npy", sums)
    

def bow(series):
    bag = {}
    for tweet in series:
        for word in tweet.split():
            if word not in bag.keys():
                bag[word] = 1
            else:
                bag[word] += 1
            
        
    return set([key for key, val in bag.items() if val > 1])


if __name__=="__main__":
    t1 = time()
    main()
    print(time()-t1)
