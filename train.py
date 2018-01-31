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

#eliminamos tags de tabulacion y signos de puntuación con esta función
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
  
       
#funcion que va a crear una lista de palabras        
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
tots0 = train.loc[(train['severe_toxic'] == 0) &
               (train['obscene'] == 0)  &
               (train['threat'] == 0) &
               (train['insult'] == 0) &  
               (train['identity_hate'] == 0) &  
               (train['toxic'] == 0) ]

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
bagtots0 = bagofwords(tots0["Processed"])



def bagofwords(train):
    wolis = baglis(list(train))

    from collections import Counter

    word_counter = Counter(wolis)

    return word_counter


wbagmain = most_common_words
wbagtox = bagofwords(toxics["Processed"])
wbagsevtox = bagofwords(severe["Processed"])
wbagobscene = bagofwords(obscene["Processed"])
wbagthreat = bagofwords(threat["Processed"])
wbaginsult = bagofwords(insult["Processed"])
wbagidentity = bagofwords(identity["Processed"])
wbagtots0 = bagofwords(tots0["Processed"])


###################
###Categoria soles##
####################

#Las siguientes bolsas de palabras son definidas solamente
#cuando esta categoria es 1 y todas las demas son iguales a 0

toxics = train.loc[(train['severe_toxic'] == 0) &
               (train['obscene'] == 0)  &
               (train['threat'] == 0) &
               (train['insult'] == 0) &  
               (train['identity_hate'] == 0) &  
               (train['toxic'] == 1) ]
severe = train.loc[(train['severe_toxic'] != 0) &
               (train['obscene'] == 0)  &
               (train['threat'] == 0) &
               (train['insult'] == 0) &  
               (train['identity_hate'] == 0)&  
               (train['toxic'] == 0) ]
obscene = train.loc[(train['severe_toxic'] == 0) &
               (train['obscene'] == 1)  &
               (train['threat'] == 0) &
               (train['insult'] == 0) &  
               (train['identity_hate'] == 0)&  
               (train['toxic'] == 0) ]
threat = train.loc[(train['severe_toxic'] == 0) &
               (train['obscene'] == 0)  &
               (train['threat'] == 1) &
               (train['insult'] == 0) &  
               (train['identity_hate'] == 0)&  
               (train['toxic'] == 0) ]
insult = train.loc[(train['severe_toxic'] == 0) &
               (train['obscene'] == 0)  &
               (train['threat'] == 0) &
               (train['insult'] == 1) &  
               (train['identity_hate'] == 0)&  
               (train['toxic'] == 0) ]
identity = train.loc[(train['severe_toxic'] == 0) &
               (train['obscene'] == 0)  &
               (train['threat'] == 0) &
               (train['insult'] == 0) &  
               (train['identity_hate'] == 1)&  
               (train['toxic'] == 0) ]


bagmain = most_common_words
bagtox = bagofwords(toxics["Processed"])
bagsevtox = bagofwords(severe["Processed"])
bagobscene = bagofwords(obscene["Processed"])
bagthreat = bagofwords(threat["Processed"])
baginsult = bagofwords(insult["Processed"])
bagidentity = bagofwords(identity["Processed"])
bagtots0 = bagofwords(tots0["Processed"])

