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
	if i in ['tripduration','from_station_id']:
		x_columns.append(i)

x=data[x_columns]
y=data['to_station_id']

temp=['start_time','gender','month','weekday','usertype','age']
for i in temp:
	newdata=data[i]
	newdata=pd.get_dummies(newdata)
	x=pd.concat([x,newdata],axis=1)

x=pd.concat([x,y],axis=1)
x.to_csv('/Users/fangjie/Desktop/datamining_project/data/dummydata.csv') #绝对位置

