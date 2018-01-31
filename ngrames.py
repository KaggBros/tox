# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:30:39 2018

@author: los40
"""
import nltk
#tagger que asigna NN a todas las palabras tokenizadas
raw = train["Processed"][4633]
default = nltk.DefaultTagger('NN') 
default.tag(raw)


patterns = [(r'.*ing$', 'VBG'),  
                # gerunds
                (r'.*ed$', 'VBD'),  
                # simple past 
                (r'.*es$', 'VBZ'),                
                # 3rd singular present 
                (r'.*ould$', 'MD'),               
                # modals 
                (r'.*\'s$', 'NN$'),               
                # possessive nouns 
                (r'.*s$', 'NNS'),                 
                # plural nouns 
                (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  
                # cardinal numbers 
                ('[t][o]', 'TO'),
                #for to
                ('[w][i][l][l]','FUTURE'),
                #future tenses                
                ('\w*an\b', "IDEN"),
                #Identity
                ('[A-Z].*[A-Z]', "AUP"),
                #any uppercase letters
                (r'.*', 'NN')                
                # nouns (default) 
                ]
                

regexp = nltk.RegexpTagger(patterns) 
regexp.tag(raw)








####PREDECIR THREAT
def bagofwords(train):
    wolis = baglis(list(train))

    from collections import Counter

    word_counter = Counter(wolis)

    most_common_words = word_counter.most_common()
    return most_common_words



th = train.loc[(train['severe_toxic'] == 0) &
               (train['obscene'] == 0)  &
               (train['threat'] == 1) &
               (train['insult'] == 0) &  
               (train['identity_hate'] == 0) ]

ths = bagofwords(th["Processed"])






