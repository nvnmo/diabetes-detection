import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn import model_selection
import pickle

df=pd.read_csv('dataset.csv')

X=df.values[:,0:8]
Y=df.values[:,8]

lsvc=Pipeline([('scaler',MinMaxScaler()),('clf',LinearSVC())])
lsvc.fit(X,Y)
pickle.dump(lsvc,open('sugr.pickle','wb'))
