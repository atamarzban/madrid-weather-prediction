#Perceptron Algorithm Trained On Madrid Weather Dataset
#Dataset Link: https://www.kaggle.com/juliansimon/weather_madrid_lemd_1997_2015.csv
#By Ata Marzban, Khatam University
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

df = df.drop(df.columns[[0, 13, 14, 15, 18, 20]], axis=1) #drop useless and too much NaN columns

subset = df.columns.values.tolist()
subset.remove(' Events') #we don't want to remove samples with NaN Events
df = df.dropna(axis=0, subset=subset) #drop the NaN samples, now we have no NaNs

y = df.iloc[:, 15]  #seperate event class
y = y.fillna('None')  #replace NaNs with Nones
df = df.drop(df.columns[[15]], axis=1) #derop event class from feature data
X = df

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=0) #train_test split

from sklearn.preprocessing import StandardScaler #scale features
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

#Train Perceptron
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=100)
ppn.fit(X_train_std, y_train) 

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f%%' % (accuracy_score(y_test, y_pred)*100)) #Calculate Accuracy
