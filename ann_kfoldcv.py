'''Created on Mon Aug  6 21:36:44 2018

@author: mukesh s
'''

# artificial neural network
# install tensorflow
# install theano
# install keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("Churn_Modelling.csv")

x=dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#print(x)

# encoding categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x_1=LabelEncoder()
x[:,1]=labelencoder_x_1.fit_transform(x[:,1])

labelencoder_x_2= LabelEncoder()
x[:,2]=labelencoder_x_2.fit_transform(x[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
#print(x[:,1])
x=onehotencoder.fit_transform(x).toarray()
x = x[:,1:]
#print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#print(x_train)
# now lets make a ANN
# importing keras libaray and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


#evaluvating,improving and tuning ann
#evaluvate ann
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()

    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))


    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))


    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics =['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn =build_classifier,batch_size=10,nb_epoch=100)
accuracies= cross_val_score(estimator=classifier, X=x_train,y=y_train,cv=10,n_jobs=-1)
mean = accuracies.mean()
variance= accuracies.std()