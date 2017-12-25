# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nltk
import csv
import pandas as pd
import os

path = "C:/Users/los40/Desktop/kaggle"
os.chdir(path)
data = pd.read_csv('./train.csv')
nltk.download("punkt")

#stop words
nltk.download("stopwords")
import string
string.punctuation
useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)

def remove_noise(row):
    """function to remove unnecessary noise from the data - sentences"""
    try:
        text = row['comment_text']
        text_splited = text.split(' ')
        text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
        noise_words = ['\n', '\n\n']
        text_splited = [''.join(c for c in s if c not in noise_words) for s in text_splited]
        text_splited = [s for s in text_splited if s]
        return(text_splited)
    except:
        return(row['comment_text'])
        
def build_bag_of_words_features_filtered(words):
    return {
        word:1 for word in words \
        if not word in useless_words}






