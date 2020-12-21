# Welcome to my series of scripts for analysis!

## Some overhead of work flow

1) Clean.py takes in a csv file and does all the cleaning of the text outlined in section 1.2.3 of the report.
2) choose.py selects the subset of tweets we are going to analyze, and it selects the test tweets for prediction later on.
3) betterDate.py converts the publish date of the tweets into a python datetime object.
4) Visualize.py takes a csv file and constructs a word cloud from the corpus of tweets.
5) analysisk.py, where k =1,2,3 are the processes by which we answer questions 1,2, and 3. Basically all three construct T matrices that are used to compare tweet via dot products.
6) results.py quantify the goodness of our predictions with confusion matrix statistics.
7) timePlot.py produces the histogram of consecutively similar tweets posted in a range of time frames.

