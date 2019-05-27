import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Model #泛型模型
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import pandas as pd

path='/Users/fangjie/Desktop/datamining_project/data/mse_data.csv'

##data process
reader=pd.read_csv(path,iterator=True)
data=reader.get_chunk(10000)
x_columns=[]
for i in data.columns:
	if i not in ['id','to_station_id']:
		x_columns.append(i)
x=data[x_columns]
y=data['to_station_id']

X_train,X_test,Y_train,Y_test=train_test_split(x,y)

X_train=np.array(X_train)
X_train=X_train.reshape(X_train.shape[0],-1)
#Y_train=np.array(Y_train)
X_test=np.array(X_test)
X_test=X_test.reshape(X_test.shape[0],-1)
#Y_test=np.array(Y_test)
print(X_train.shape)
print(X_test.shape)


##encoded dimension
encoding_dim=3

##input
input_data=Input(shape=(8,))

##encoded
encoded=Dense(6,activation='relu')(input_data)
encoded=Dense(4,activation='relu')(encoded)
encoder_output=Dense(encoding_dim)(encoded)

##decoded
decoded=Dense(4,activation='relu')(encoder_output)
decoded=Dense(6,activation='relu')(decoded)
decoded=Dense(8,activation='tanh')(decoded)

##autoencoder model
autoencoder=Model(input=input_data,output=decoded)

##encoded model
encoder=Model(input=input_data,output=encoder_output)

#compile 
autoencoder.compile(optimizer='adam',loss='mse')

#training
autoencoder.fit(X_train,X_train,epochs=30,batch_size=64,shuffle=True)
t=autoencoder.evaluate(X_test,X_test)
print(t)











