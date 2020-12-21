#!/usr/bin/env python
## analysis functions & libraries

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import mutual_info_classif, SelectKBest

## got help with Vectorizers from this article:
## https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af
import spacy
from html import unescape


class MyTfidfVectorizer(TfidfVectorizer):

    # override the build_analyser
    def build_analyzer(self):
        '''get stop words from this method inherited from TfidfVectorizer'''
        stop_words = self.get_stop_words()

        def analyser(doc):
            '''analyzer returned from build_analyzer'''
            try:
                spacy.load('en')
                lemmatizer = spacy.lang.en.English()
                doc_clean = unescape(doc).lower()
                tokens = lemmatizer(doc_clean)
                lemmatized_tokens = [token.lemma_ for token in tokens]
                ## calls inherited method that removes stop words and builds ngram
                return self._word_ngrams(lemmatized_tokens, stop_words)
            except TypeError:
                print(doc)

        return analyser


class MyCountVectorizer(CountVectorizer):

    # override the build_analyser
    def build_analyzer(self):
        '''get stop words from this method inherited from CountVectorizer'''
        stop_words = self.get_stop_words()

        def analyser(doc):
            '''analyzer returned from build_analyzer'''
            try:
                spacy.load('en')
                lemmatizer = spacy.lang.en.English()
                doc_clean = unescape(doc).lower()
                tokens = lemmatizer(doc_clean)
                lemmatized_tokens = [token.lemma_ for token in tokens]
                ## calls inherited method that removes stop words and builds ngram
                return self._word_ngrams(lemmatized_tokens, stop_words)
            except TypeError:
                print(doc)

        return analyser
