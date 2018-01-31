# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:58:01 2018

@author: los40
"""

import itertools
from numpy import where
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pandas as pd
import os
from sklearn.metrics import accuracy_score


df = data.copy()

y=df.loc[:,'toxic':'identity_hate']

x_train, x_test, y_train, y_test = train_test_split(df['comment_text'],
                                                    y, test_size=0.30,
                                                    random_state=53)

count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(x_train.values)
count_test = count_vectorizer.transform(x_test.values)
pred_test = OneVsRestClassifier(LinearSVC(random_state=0)).fit(count_train, y_train).predict(count_test)

#comprobamos la efectividad del modelo
pred_testd = pd.DataFrame(pred_test, columns=list(data.columns.values)[2:len(data.columns.values)])
cols = list(data.columns.values)[2:len(data.columns.values)]
from mlxtend.evaluate import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

plt.subplot(3,3,3) 
for i in range(len(cols)):
    cm = confusion_matrix(y_target=y_test[cols[i]], 
                      y_predicted=pred_testd[cols[i]])
    
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    plt.title(cols[i])
    plt.show()


for i in range(len(cols)):
   print( accuracy_score(y_test[cols[i]],pred_testd[cols[i]]))


for j in range(len(cols)):  
    uns = [i for i, v in enumerate(y_test[cols[j]]) if v == 1]
    ok= sum(pred_testd[cols[j]][uns]==1)/sum(y_test[cols[j]]==1)
    print(cols[j])
    print(ok)




accuracy_score(y_test[cols[1]][y_test[cols[1]]==1],pred_testd[cols[1]][uns])


#################
##############
#########
#####
###
#



onevsrest = OneVsRestClassifier(LinearSVC(random_state=0)).fit(count_train, y_train)
pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(count_train, y_train).predict(count_test)


test0 = pd.read_csv("C:/Users/los40/Desktop/kaggle/test.csv")
test1 = test0['comment_text']

count_test2 = count_vectorizer.transform(test1.values)
predTestSet = onevsrest.predict(count_test2)
pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(count_train, y_train).predict(count_test2)




#crear tabla per enviar
predic = pd.DataFrame(data=pred, columns=list(data.columns.values)[2:len(data.columns.values)])
test0['id']

final = pd.concat([test0['id'], predic], axis=1)

os.chdir("C:/Users/los40/Desktop/kaggle/")

final.to_csv('final.csv',sep=',', encoding='utf-8')
