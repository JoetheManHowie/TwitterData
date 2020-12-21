#!/usr/bin/env python

## ex: getVisual.py <tweets.csv> <savefilename>

from datetime import datetime as dt
import sys
import pandas as pd
import numpy as np
#import nltk
from time import time
import warnings

# for visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 200

def main():
    csv_in = sys.argv[1]
    savefile = sys.argv[2]
    print("Importing data : ", dt.now())
    df = import_df(csv_in)
    print("Creating viualization : ", dt.now())
    visualize_common_words(df, savefile)


def import_df(csv_in):
    return pd.read_csv(csv_in, dtype={"author": str,
                                      "content": str,
                                      "account_type": str,
                                      "publish_date": str})


def visualize_common_words(df, savefile):
    bow = {} ## keys words: values: frequency
    tweets = df.content.str.split()
    these_tweets = []
    count = 0
    for words in tweets:
        for word in words:
            if word not in bow.keys():
                bow[word] =1
            else:
                bow[word] +=1



    wordcloud = WordCloud(width=6000,
                          height=4000,
                          random_state=1,
                          background_color='cornflowerblue',
                          collocations=False,
                          stopwords=STOPWORDS).generate_from_frequencies(bow)

    plt.figure()
    plt.imshow(wordcloud)#, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(savefile+".jpg")
    plt.close()


if __name__=='__main__':
    t1 = time()
    main()
    print(dt.now(), time()-t1)
