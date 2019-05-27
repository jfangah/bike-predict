import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


path='/Users/fangjie/Desktop/datamining_project/data/preprocessed_data.csv'

#data 
#reader=pd.read_csv(path,iterator=True)
#data=reader.get_chunk(10000)
data=pd.read_csv(path)
x_columns=[]
for i in data.columns:
	if i not in ['id','to_station_id']:
		x_columns.append(i)
x=data[x_columns]
y=data['to_station_id']

X_train,X_test,Y_train,Y_test=train_test_split(x,y)

#model
X_train=X_train.values
X_test=X_test.values
Y_train=Y_train.values
Y_test=Y_test.values
gbr=GradientBoostingClassifier()
gbr.fit(X_train,Y_train)

y_gbr=gbr.predict(X_train)
y_gbr1=gbr.predict(X_test)
acc_train = gbr.score(X_train, Y_train)
acc_test = gbr.score(X_test, Y_test)
print(acc_train)
print(acc_test)