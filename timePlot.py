#!/usr/bin/env python

from time import time
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200

def main():
    print("---- load data ----", dt.now())
    right = make_occ(np.load("Right_q3ans.npy", allow_pickle=True))
    left  = make_occ(np.load("Left_q3ans.npy", allow_pickle=True))
    local = make_occ(np.load("local_q3ans.npy", allow_pickle=True))
    news  = make_occ(np.load("news_q3ans.npy", allow_pickle=True))
    
    bin_n = [a for a in range(0,130,5)]
    ### plot ###
    print('---- make plot ----', dt.now())
    plt.figure()
    plt.hist(right, bin_n, log=True,label="Right", histtype='step', density=True,stacked=True, fill=False,color='blue')
    plt.hist(left, bin_n, log=True,label="Left", histtype='step', density=True,stacked=True, fill=False,color='red')
    plt.hist(local, bin_n, log=True,label="local", histtype='step', density=True,stacked=True, fill=False,color='black')
    plt.hist(news, bin_n, log=True,label="news", histtype='step', density=True,stacked=True, fill=False, color='magenta')
    plt.legend(loc = "best")
    plt.xlabel("Time difference (minutes)")
    plt.ylabel("Normalized number of related tweets")
    plt.title("Number of tweets related by more than 0.25\nvs the time span over which tweets were posted")
    plt.xlim([0, 125])
    plt.savefig("q3Plot.jpg")
    plt.close()


def make_occ(arr):
    ans = np.array([])
    for ti, num in arr:
        ans = np.append(ans, np.array([ti]*int(num-1)))
    return(ans)
    
    
if __name__=="__main__":
    t1 = time()
    main()
    print(time()-t1)

