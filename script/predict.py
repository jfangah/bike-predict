import pandas as pd
from math import isnan
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math


def my_predict(train_data_path,test_data_path):
	##train_data
	training_data = pd.read_csv(train_data_path,parse_dates=['start_time'], date_parser=lambda col: pd.to_datetime(col))
	training_data = training_data.drop(training_data.columns[0], axis=1)

	# delete the useless data
	training_data.drop(['end_time', 'bikeid', 'from_station_name', 'to_station_name'],axis = 1,inplace = True)

	# Change the types of data.
	
	training_data['usertype'] = training_data['usertype'].map(lambda x: -1 if x == 'Customer' else 1)
	training_data['gender'] = training_data['gender'].map(lambda x: 1 if x == 'Male' else (-1 if x == 'Female' else 0))
	training_data['birthyear'] = training_data['birthyear'].map(lambda x: int((2018 - x - 1) / 5) if isnan(x)is not True else 0)
	training_data['age'] = training_data['birthyear'].map(lambda x: x if x >= 0 and x <= 15 else 16)
	training_data['month'] = training_data['start_time'].dt.month
	training_data['weekday'] = training_data['start_time'].dt.dayofweek+1
	training_data['start_time'] = training_data['start_time'].dt.hour
	training_data['tripduration'] = training_data['tripduration'].map(lambda x: int(float(x.strip().replace(',', '')) / 60))
	training_data.drop(['birthyear'],axis = 1,inplace = True)


	##knn
	data=training_data
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
	print('the train accuracy is ',accuracy)


	##test data
	test_data = pd.read_csv(test_data_path, parse_dates=['start_time'], date_parser=lambda col: pd.to_datetime(col))
	test_data = test_data.drop(test_data.columns[0], axis=1)

	# delete the useless data
	test_data.drop(['end_time', 'bikeid', 'from_station_name', 'to_station_id', 'to_station_name'],axis = 1,inplace = True)

	test_data['usertype'] = test_data['usertype'].map(lambda x: -1 if x == 'Customer' else 1)
	test_data['gender'] = test_data['gender'].map(lambda x: 1 if x == 'Male' else (-1 if x == 'Female' else 0))
	test_data['birthyear'] = test_data['birthyear'].map(lambda x: int((2018 - x - 1) / 5) if isnan(x)is not True else 0)
	test_data['age'] = test_data['birthyear'].map(lambda x: x if x >= 0 and x <= 15 else 16)
	test_data['month'] = test_data['start_time'].dt.month
	test_data['weekday'] = test_data['start_time'].dt.dayofweek+1
	test_data['start_time'] = test_data['start_time'].dt.hour
	test_data['tripduration'] = test_data['tripduration'].map(lambda x: int(float(x.strip().replace(',', '')) / 60))
	test_data.drop(['birthyear'],axis = 1,inplace = True)



	##predict
	test_columns=[]
	for i in test_data.columns:
	    test_columns.append(i)
	test_columns.append('from_station_id')
	test_columns.append('from_station_id')
	test_columns.append('from_station_id')
	test_columns.append('start_time')
	test_columns.append('start_time')
	test_columns.append('usertype')
	test_data=test_data[x_columns]

	test_predicted=estimator.predict(test_data)
	to_df = pd.DataFrame(test_predicted)
	to_df.columns = ['to_station_id']
	return to_df



def main():
	train_data_path = 'data/bikeshareTraining.csv'
	test_data_path = 'data/bikeshareTest.csv'
	predict_value = my_predict(train_data_path,test_data_path)
	print(predict_value)

if __name__=="__main__":
	main()