# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:58:01 2018

@author: los40
"""


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy import interp

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




#accuracy_score(y_test[cols[1]][y_test[cols[1]]==1],pred_testd[cols[1]][uns])


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

ok = test0['id']
for i in range(len(test0['id'])):
    ok[i]= "'" + test0['id'][i]
    


#final = pd.concat([ok, predic], axis=1)
#os.chdir("C:/Users/los40/Desktop/kaggle/")
#final.to_csv('final.csv',sep=',', encoding='utf-8')

### ROC CURVE FOR THE ONEVSREST CLASSIFIER (LINEAR SVM)


###trying to add validation set
#sample_weight = np.random.RandomState(42).rand(y.shape[0])
#xx_train, xx_test, yy_train, yy_test, sw_train, sw_test = train_test_split(df['comment_text'],y,
                                #        sample_weight, test_size=0.3, random_state=42)

#ccount_train = count_vectorizer.fit_transform(xx_train.values)
#ccount_test = count_vectorizer.transform(xx_test.values)
#swcount_test = count_vectorizer.transform(sw_test.values)

######
#pred1 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(count_train, y_train).predict(count_test)
#pred1= pd.DataFrame(data=pred1, columns=list(data.columns.values)[2:len(data.columns.values)])



##PROBABILITY CALIBRATION

from sklearn.calibration import CalibratedClassifierCV
onelsvm = LinearSVC(random_state=0)
onelsvm_c = CalibratedClassifierCV(onelsvm, method='sigmoid') 
onelsvm_c_o = OneVsRestClassifier(onelsvm_c)
olsvm_c_o_f=onelsvm_c_o.fit(count_train, y_train)

prob_svmo = olsvm_c_o_f.predict_proba(count_test)[:, 1]
prob_svmo_t = olsvm_c_o_f.predict_proba(count_test)

from sklearn.metrics import brier_score_loss
for i in range(6):
    clf_sigmoid_score = brier_score_loss(y_test.iloc[:,i], prob_svmo_t[:,i],sample_weight=None )
    print("With sigmoid calibration:" ,cols[i], clf_sigmoid_score)

onelsvm_is = CalibratedClassifierCV(onelsvm, method='isotonic') 
onelsvm_is = OneVsRestClassifier(onelsvm_is)
olsvm_is_f=onelsvm_c_o.fit(count_train, y_train)

prob_is_t = olsvm_is_f.predict_proba(count_test)

for i in range(6):
    clf_isotonic_score = brier_score_loss(y_test.iloc[:,i], prob_is_t[:,i],sample_weight=None )
    print("With isotonic calibration:" ,cols[i], clf_sigmoid_score)
    
    
prob_is_t
prob_svmo_t
probis = pd.DataFrame(data=prob_is_t, columns=list(data.columns.values)[2:len(data.columns.values)])
prosig = pd.DataFrame(data=prob_svmo_t, columns=list(data.columns.values)[2:len(data.columns.values)])


probisnd = probis.values
prosignd = prosig.values

roc_macro(roc(y_test, probisnd))
roc_macro(roc(y_test, prosignd))

probis.iloc[:,0]
probis.iloc[:,1]
prosig.iloc[:,2]
prosig.iloc[:,3]
prosig.iloc[:,4]
prosig.iloc[:,5]

probisig = pd.DataFrame({cols[0]: probis.iloc[:,0], cols[1]: probis.iloc[:,1]
, cols[2]: prosig.iloc[:,2], cols[3]: prosig.iloc[:,3], cols[4]: prosig.iloc[:,4], cols[5]: probis.iloc[:,5]})
probisignd = probisig.values

roc_macro(roc(y_test, probisignd))


#prob_is_t_ex = olsvm_is_f.predict_proba(count_test2)
#probis_ex = pd.DataFrame(data=prob_is_t_ex, columns=list(data.columns.values)[2:len(data.columns.values)])




#final_lsvm_iso = pd.concat([probis_ex], axis=1)
#os.chdir("C:/Users/los40/Desktop/kaggle/")
#final_lsvm_iso.to_csv('final_lsvm_iso.csv',sep=',', encoding='utf-8')


#######


n_classes = 6
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test.iloc[:, i], pred1.iloc[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
y_test1 = y_test.values
y_score = pred1.values
lw=2

fpr["micro"], tpr["micro"], _ = roc_curve(y_test1.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

from itertools import cycle

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
    
###GUARDANDO EL FICHERO

df = tots0['id']

f = open("file.txt", "w")
f.write("\n".join(map(lambda x: str(x), list(ok))))
f.close()
    
    
###MULTILABEL CLASSIFIERS - TREES

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()    
dtc_m =clf.fit(count_train, y_train)
pred_t = dtc_m.predict(count_test2)

pred_t = pd.DataFrame(data=pred_t, columns=list(data.columns.values)[2:len(data.columns.values)])
#final_t = pd.concat([pred_t], axis=1)
#os.chdir("C:/Users/los40/Desktop/kaggle/")
#final_t.to_csv('final_t.csv',sep=',', encoding='utf-8')



####NEURAL NETWORK
## 0.67 AUC (KAGGLE)
from sklearn.neural_network import MLPClassifier

nnet = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
nnet_m =nnet.fit(count_train, y_train)
pred_n = nnet_m.predict(count_test2)

pred_n_pd = pd.DataFrame(data=pred_n, columns=list(data.columns.values)[2:len(data.columns.values)])
final_n = pd.concat([pred_n_pd], axis=1)
os.chdir("C:/Users/los40/Desktop/kaggle/")
final_n.to_csv('final_n.csv',sep=',', encoding='utf-8')


##ROC
n_classes = 6
fpr_n = dict()
tpr_n = dict()
roc_auc_n = dict()
for i in range(n_classes):
    fpr_n[i], tpr_n[i], _ = roc_curve(y_test1[:, i], pred_n[:, i])
    roc_auc_n[i] = auc(fpr[i], tpr[i])



### RANDOM FOREST
    
from sklearn.ensemble import RandomForestClassifier
RANDOM_STATE = 123
rfm =  RandomForestClassifier(warm_start=True, max_features=None,oob_score=True, random_state=RANDOM_STATE)
#rfm_fit=rfm.fit(count_train, y_train)
pred_rf = rfm_fit.predict(count_test)
pred_rf_pd = pd.DataFrame(data=pred_rf, columns=list(data.columns.values)[2:len(data.columns.values)])

pred_rf_t = rfm_fit.predict(count_test2)
pred_rf_pd = pd.DataFrame(data=pred_rf_t, columns=list(data.columns.values)[2:len(data.columns.values)])

n_classes = 6
def roc(ytest, prediction):
    fpr_t = dict()
    tpr_t = dict()
    roc_auc_t = dict()
    for i in range(n_classes):
        fpr_t[i], tpr_t[i], _ = roc_curve(ytest.iloc[:, i], prediction[:, i])
        roc_auc_t[i] = auc(fpr_t[i], tpr_t[i])
    return(roc_auc_t)

def roc_macro(roc_aucs):
    suma=0
    for i in range(n_classes):
        suma = suma + roc_aucs[i]
    return(suma/n_classes)
    
roc_macro(roc(y_test, pred_rf))




#final_rf = pd.concat([pred_rf_pd], axis=1)
#os.chdir("C:/Users/los40/Desktop/kaggle/")
#final_rf.to_csv('final_rf.csv',sep=',', encoding='utf-8')

