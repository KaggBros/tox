# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 19:50:35 2018

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


#df = data.copy()
#dfTrain = data.copy()
#df['class']=where(dfTrain.identity_hate==1, "identity_hate", 
 #      where(dfTrain.threat==1, "threat",
          #   where(dfTrain.insult==1, "insult",
           #        where(dfTrain.obscene==1, "obscene",
            #             where(dfTrain.severe_toxic==1, "severe_toxic",
             #                  where(dfTrain.toxic==1, "toxic", 0))))))


####Feature Engineering

#df['Future']



#creamos variable respuesta
y = df['class']

#dividimos las explicativas y respuestas con un ratio 30/70, entre test set
#y set de entrenamiento
x_train, x_test, y_train, y_test = train_test_split(df['comment_text'],
                                                    y, test_size=0.30,
                                                    random_state=53)



#tokeniza las palabras y las cuenta, como las bag of word que hemos creado
#en train.py
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(x_train.values)
count_test = count_vectorizer.transform(x_test.values)


#creacion del modelo
nb_classifier = MultinomialNB()
nb_classifier.fit(count_train, y_train)
#almacenamos predicciones
pred = nb_classifier.predict(count_test)





#Analisis del model predictiu:

#comprobamos la efectividad del modelo
metrics.accuracy_score(y_test, pred)
metrics.confusion_matrix(y_test, pred)


#funcio per confusion matrix:

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#confusion matrix plot

cnf_matrix = metrics.confusion_matrix(y_test, pred)
class_names = ['no_toxic','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


plt.savefig('matriz_confusion'+'.png')
plt.show()




####probar en el test


import pandas as pd


test1 = pd.read_csv("C:/Users/los40/Desktop/kaggle/test.csv")
#test["Processed"] = test.apply(lambda row: remove_noise(row), axis = 1)
test1 = test1['comment_text']
#test1.dropna(test1, axis = 1, how = 'all')
test2 = test1[pd.notnull(test1)]

count_vectorizer = CountVectorizer(stop_words='english')

#v = TfidfVectorizer(decode_error='replace', encoding='utf-8')
count_test2 = count_vectorizer.transform(test2.values)
predTestSet = nb_classifier.predict(count_test2)


test_sub = pd.read_csv("C:/Users/los40/Desktop/kaggle/test.csv")
test_sub = test_sub['comment_text']
test_sub = test_sub[pd.notnull(test_sub)]
test_sub["pred"] = predTestSet


##############################################
##############################################
dfTrain.loc[1, 'tokenized']
dfTrain1=dfTrain.loc[0:1,]
dfTrain1.loc['tokenized_clean']=dfTrain1.apply(lambda row: stop_word_remover(row['tokenized']), axis=1)

sampleSent=dfTrain.loc[0, 'comment_text']
tokenized=nltk.word_tokenize(sampleSent)


word_list=dfTrain.loc[0, 'tokenized']

print(stopwords)

 
#########################################
####LOGISTIC REGRESSION MULTICLASS ######
#########################################
from sklearn import linear_model, datasets
logreg = linear_model.LogisticRegression(C=1e5, multi_class = 'ovr' )
dfl = data.copy()
yy = dfl.loc[:, 'toxic':'identity_hate']


#dfl["Processed"] = dfl.apply(lambda row: remove_noise(row), axis = 1)
xx_train, xx_test, yy_train, yy_test = train_test_split(dfl['comment_text'],
                                                    yy, test_size=0.30,
                                                    random_state=53)

tvect = TfidfVectorizer(min_df=1, max_df=1, stop_words = 'english')
ccount_train = tvect.fit_transform(xx_train.values)

#ccount_vectorizer = CountVectorizer(stop_words='english')
#ccount_train = ccount_vectorizer.fit_transform(xx_train.values)
#ccount_test = ccount_vectorizer.transform(xx_test.values)



logreg.fit(ccount_train, yy_train)






#####################
######
######################

from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
#pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(ccount_train, yy_train).predict(count_test)
#metrics.accuracy_score(y_test, pred)

onevsrest = OneVsRestClassifier(LinearSVC(random_state=0)).fit(ccount_train, yy_train)
pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(ccount_train, yy_train).predict(count_test)




import pandas as pd
test1 = pd.read_csv("C:/Users/los40/Desktop/kaggle/test.csv")
#test["Processed"] = test.apply(lambda row: remove_noise(row), axis = 1)
test1 = test1['comment_text']
#test1.dropna(test1, axis = 1, how = 'all')
test2 = test1[pd.notnull(test1)]

#count_vectorizer = CountVectorizer(stop_words='english')


#v = TfidfVectorizer(decode_error='replace', encoding='utf-8')
count_test2 = tvect.fit_transform(test2.values)
predTestSet = onevsrest.predict(count_test2)
pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(ccount_train, yy_train).predict(count_test2)


test2['preds'] = predTestSet

import numpy as np

np.savetxt('trainee.csv', predTestSet, fmt='%d', delimiter=',')