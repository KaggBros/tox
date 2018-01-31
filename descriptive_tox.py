# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:58:52 2018

@author: los40
"""

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import savefig 


import os
path = "C:/Users/los40/Desktop/kaggle/descriptiva"
os.chdir(path)
#data = data.copy()
tipos = ["threat", "insult", "obscene", "severe_toxic", "identity_hate" ]
for i in tipos:
 data_cat = data[data[i]==1]
 threat_text = "".join(data_cat['comment_text'])
 wordcloud = WordCloud(collocations =False).generate(threat_text)
 plt.imshow(wordcloud)
 plt.axis('off')
 plt.savefig(i + 'WORDCLOUD' +'.png')
 plt.show()
 
 
##solo esta categoria
tipos = ["threat", "insult", "obscene", "severe_toxic", "identity_hate" ]
for i in tipos:
    data_cat = data[data[i]==1]
    threat_text = "".join(data_cat['comment_text'])
    wordcloud = WordCloud(collocations =False).generate(threat_text)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig(i + 'WORDCLOUD' +'soles'+'.png')
    plt.show()
 


#grafico logaritmo conteo palabras

def plotlog(wbag, label):
 sorted_word_counts = sorted(list(wbag.values()), reverse=True)
 plt.loglog(sorted_word_counts ,label = label)
 plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

 
#falta añadir leyenda por colores
#explicacion del grafico 
plotlog(wbagtox, 'tox')
plotlog(wbagsevtox, 'sev_tox')
plotlog(wbagobscene, 'obscene')
plotlog(wbagthreat, 'threat')
plotlog(wbaginsult, 'insult')
plotlog(wbagidentity, 'identity_hate')
plotlog(wbagtots0, 'todos')
plt.ylabel("Freq")
plt.xlabel("Word Rank")
plt.savefig('loglog1.png')
plt.show()







#Toxicidad total en cada comentario
datad = data.copy()
selCols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
datad['overallToxicity'] = data[selCols].sum(axis = 1)
#incluyendo los 0
datad['overallToxicity'].plot(kind='hist')
plt.savefig('toxicidad_total_con0'+'.png')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
 
#excluyendo los 0
datad[datad['overallToxicity'] != 0]['overallToxicity'].plot(kind='hist')
plt.savefig('toxicidad_total_sin0'+'.png')
plt.show()
 

sum(datad['overallToxicity']>0)/len(datad['overallToxicity'])

