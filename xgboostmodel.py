# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:34:14 2018

@author: los40
"""

import xgboost as xgb
from xgboost.sklearn import XGBClassifier  


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

xdf = data.copy()

y=xdf.loc[:,'toxic':'identity_hate']

x_train, x_test, y_train, y_test = train_test_split(xdf['comment_text'],
                                                    y, test_size=0.30,
                                                    random_state=53)





