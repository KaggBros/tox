# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd

path = "C:/Users/los40/Desktop/kaggle"
os.chdir(path)
import nltk

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
        bag = {}
        for i in len(words):
            bag.update = {   word:1 for word in words[i]  if not word in useless_words}
        
def baglis(words):
        bag = []
        for i in range(len(words)):
            for j in range(len(words[i])):
                if words[i][j] not in useless_words:
                     bag.append(  words[i][j])
        return (bag)
    
            
        
train = data.copy()
train["Processed"] = train.apply(lambda row: remove_noise(row), axis = 1)
wolis = baglis(train["Processed"])

from collections import Counter

word_counter = Counter(wolis)

most_common_words = word_counter.most_common()

toxics = train['toxic']> 0
sev = train['severe_toxic']> 0
obs = train['obscene']> 0
thr = train['threat']> 0
ins = train['insult']> 0
ide = train['identity_hate']> 0

toxics = train[toxics]
severe = train[sev]
obscene = train[obs]
threat = train[thr]
insult = train[ins]
identity = train[ide]


def bagofwords(train):
    wolis = baglis(list(train))

    from collections import Counter

    word_counter = Counter(wolis)

    most_common_words = word_counter.most_common()
    return most_common_words

bagmain = most_common_words
bagtox = bagofwords(toxics["Processed"])
bagsevtox = bagofwords(severe["Processed"])
bagobscene = bagofwords(obscene["Processed"])
bagthreat = bagofwords(threat["Processed"])
baginsult = bagofwords(insult["Processed"])
bagidentity = bagofwords(identity["Processed"])






