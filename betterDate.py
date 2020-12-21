#!/usr/bin/env python

## ex: betterDates.py <file.csv>

##output <dated_file.csv>

from datetime import datetime as dt
from datetime import date
from datetime import time as Time
from time import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi']=200

def main():
    csv_in = sys.argv[1]
    savefile = csv_in.split(sep=".")[0]
    df =  pd.read_csv(csv_in)
    df = convert_datetime(df)
    df.to_csv("dated_"+savefile+".csv", index=False)


def convert_datetime(df):
    df[["date", "time"]] = df.publish_date.str.split(expand=True)
    df[['hour', 'minute']] = df.time.str.split(pat=":", expand=True)
    df[['month', 'day', 'year']] = df.date.str.split(pat="/", expand=True)
    df.hour = df.hour.astype(int)
    df.minute = df.minute.astype(int)
    df.year = df.year.astype(int)
    df.month = df.month.astype(int)
    df.day = df.day.astype(int)
    df["datetime"] = df[['year', 'month', 'day', 'hour', 'minute']].apply(lambda s: dt(*s), axis=1)
    df.drop(["publish_date", 'year', "month", 'day', 'hour', 'minute', "date", "time"], axis=1, inplace=True)
    return df
    

if __name__=="__main__":
    t1 = time()
    main()
    print(time()-t1)
