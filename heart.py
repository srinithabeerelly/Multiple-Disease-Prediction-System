import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#%% reading Dataset
df = pd.read_csv("heart.csv")
#%% looking at dataset
df.head()
df.isnull().sum()
df.describe()
df['target'].value_counts() # 1 Defective heart 0 Healthy heart

x = df.drop(columns='target', axis=1)
y = df['target']
#%% training
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.1,random_state=42)
model = LogisticRegression()
model.fit(x_train,y_train)

#%% Model Evaluation

x_training_pred = model.predict(x_train)
training_data_acc = accuracy_score(x_training_pred,y_train)
print(f"Accuracy on training data : {training_data_acc}")

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print(f"Accuracy on test data : {test_data_accuracy}")

#%%

input_data = (70,5,2,111,200,0,0,120,0,1.2,10,2,2)
input_data_as_numpy_array= np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
#%%
import pickle
filename = 'diabets.sav'
pickle.dump(model, open(filename, 'wb'))