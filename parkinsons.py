import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

#%% Loading and looking for null datas
df = pd.read_csv("parkinsons.csv")
df.isnull().sum()

#%%
df.head()
# distribution of target Variable
df['status'].value_counts()

# group the data based on the target variable
df.groupby('status').mean()

x = df.drop(columns=['name','status'], axis=1)
y = df['status']

#%% Training Data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#%% Model Training 
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

#%% Model Evaluation

x_train_pred = model.predict(x_train)
x_train_acc = accuracy_score(x_train_pred,y_train)
print(f"Accuracy score of train data: {x_train_acc}")


x_test_pred = model.predict(x_test)
x_test_acc = accuracy_score(x_test_pred,y_test)
print(f"Accuracy score of train data: {x_test_acc}")

#%%  Building a Predictive System
input_data = (203.07600,226.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")
  #%%
import pickle
filename = 'parkinsons.sav'
pickle.dump(model, open(filename, 'wb'))