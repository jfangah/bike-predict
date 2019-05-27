import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import math




path='/Users/fangjie/Desktop/datamining_project/data/preprocessed_data.csv'
#path='/Users/fangjie/Desktop/datamining_project/data/mse_data.csv'

##data process

data=pd.read_csv(path)
x_columns=[]
for i in data.columns:
	if i not in ['id','to_station_id']:
		x_columns.append(i)
x_columns.append('from_station_id')
x_columns.append('from_station_id')
x_columns.append('from_station_id')
x_columns.append('start_time')
x_columns.append('start_time')
x_columns.append('usertype')
x=data[x_columns]
y=data['to_station_id']

X_train,X_test,Y_train,Y_test=train_test_split(x,y)
X_train=X_train.values
X_test=X_test.values
Y_train=Y_train.values
Y_test=Y_test.values

##knn model
estimator=KNeighborsClassifier(n_neighbors=5,weights='distance',algorithm='kd_tree')
estimator.fit(X_train,Y_train)

y_predicted=estimator.predict(X_test)

accuracy=np.mean(Y_test==y_predicted)*100
print(accuracy)