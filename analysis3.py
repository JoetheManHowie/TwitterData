#!/usr/bin/env python

## ex: analysis3.py <file.csv>

from datetime import datetime as dt
from time import time
import sys
import pandas as pd
import numpy as np
from MyVecs import MyTfidfVectorizer


def main():
    csv = sys.argv[1]
    savefile = csv.split(sep="_")[0]+"_q3ans"
    #### importing data ####
    print("---- importing data ----", dt.now())
    df = pd.read_csv(csv, parse_dates=["datetime"])
    #### vectorizing data ####
    df = df.loc[df.author==df.author.unique()[0]] # take only top user
    print("---- vectorizing data ----", dt.now())
    vector = MyTfidfVectorizer(ngram_range=(1,2),
                               stop_words='english',
                               vocabulary=bow(df.content))
    print("vocab:", len(vector.get_feature_names()))
    #### fit vector to docs ####
    print("---- fitting data to transform ----", dt.now())
    tfidf = vector.fit_transform(df.content)
    #### matrix multiplication ####
    print("---- matrix multiplication ----", dt.now())
    dots = tfidf.dot(tfidf.T)
    part = get_partitions(df.datetime, dots)
    #df.to_csv(savefile+".csv", index=False)
    output = np.empty([0,2])
    for i, j in part:
        num = j+1-i
        tweets = df.content.iloc[range(i, j+1)] 
        time_diff = (df.datetime.iloc[j]-df.datetime.iloc[i]).total_seconds()/60
        output = np.vstack((output, np.array([time_diff, num])))
        print(tweets)
        print(time_diff)
        print()


    #### save data .npy ####
    print("---- save data .npy ----", dt.now())
    np.save(savefile+".npy", output)
    

def get_partitions(timestamps, dots):
    part = []
    st = 0
    ed = 0
    for i in range(len(timestamps)-1):
        if dots[i, i+1] < 0.25:
            ed = i
            if ed != st:  
                part.append((st,ed))
            st = i+1

    return part    


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
