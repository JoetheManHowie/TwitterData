#!/usr/bin/env python

## ex: select.py <file.csv>

## output: Right.csv, Left.csv, local.csv, news.csv

from time import time
from datetime import datetime as dt
import sys
import pandas as pd
import numpy as np

def main():
    csv = sys.argv[1]
    #### import data ####
    print("---- import data ----", dt.now())
    df = pd.read_csv(csv, parse_dates=["datetime"])
    df.sort_values(by=["datetime"], inplace=True)
    #### partition data ####
    print("---- partition data ----", dt.now())

    selec(df.loc[df.account_type=="Right"], dt(2017, 6, 10), dt(2017, 6, 30)).to_csv("Right.csv", index=False)
    right = selec(df.loc[df.account_type=="Right"], dt(2017, 7,  2), dt(2017, 7, 8))#.to_csv("Right_test.csv", index=False)
    
    selec(df.loc[df.account_type=="Left"],  dt(2017, 6, 30), dt(2017, 9, 30)).to_csv("Left.csv",  index=False)
    left = selec(df.loc[df.account_type=="Left"],  dt(2017, 10, 1), dt(2017, 11, 25))#.to_csv("Left_test.csv",  index=False)

    selec(df.loc[df.account_type=="local"], dt(2017, 6, 14), dt(2017, 6, 20)).to_csv("local.csv", index=False)
    local = selec(df.loc[df.account_type=="local"], dt(2017, 6, 21), dt(2017, 6, 22))#.to_csv("local_test.csv", index=False)

    selec(df.loc[df.account_type=="news"],  dt(2017, 5, 15), dt(2017, 11,15)).to_csv("news.csv",  index=False)
    news = selec(df.loc[df.account_type=="news"],  dt(2017, 1, 20), dt(2017, 2, 25))#.to_csv("news_test.csv",  index=False)
    print(np.shape(right))
    print(np.shape(left))
    print(np.shape(local))
    print(np.shape(news))
    ans = right.append(left.append(local.append(news)))
    ans.to_csv('test_file.csv', index=False)
    print("---- Done ----", dt.now())


def selec(df, low, high):
    limit = 20
    df = df.loc[df.author.isin(df.author.value_counts()[:limit].index)]
    df = df.loc[(df.datetime > low) & (df.datetime < high)]
    return(df)
    


if __name__=="__main__":
    t1 = time()
    main()
    print(time()-t1)
