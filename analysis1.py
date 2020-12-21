#!/usr/bin/env python

## ex analysis1.py <file.csv> <limit>

from datetime import datetime as dt
from time import time
import sys
import pandas as pd
import numpy as np
from MyVecs import MyTfidfVectorizer
from scipy.sparse import find

def main():
    csv = sys.argv[1]
    savefile = csv.split(sep=".")[0]
    limit = float(sys.argv[2])
    #### import data ####
    print("---- importing tweet data ----", dt.now())
    df = pd.read_csv(csv)
    df.dropna(inplace=True)
    print([index for index, row in df.iterrows() if row.isnull().any()])
    #### make vector ####
    print("---- make vector ----", dt.now())
    vector = MyTfidfVectorizer(ngram_range=(1, 2),
                               stop_words='english',
                               vocabulary=bow(df.content))
    #### fit vector to docs ####
    print("vocab:", len(vector.get_feature_names()))
    print("---- fitting tweets to vector ----", dt.now())
    tfidf = vector.fit_transform(df.content)
    #### calculate dots products ####
    print("---- calculating dot products ----", dt.now())
    dots = tfidf.dot(tfidf.T)
    ind = set()
    for i,j,_ in zip(*find(dots> limit)):
        ind.add(i)
        ind.add(j)
    ind = list(ind)
    #### save similar tweets for visualization ####
    print("---- saving csv ----", dt.now())
    df.iloc[ind].to_csv(savefile+"_q1ans.csv", index=False)


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
