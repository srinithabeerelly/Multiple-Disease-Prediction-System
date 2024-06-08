import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#%%
df = pd.read_csv("diabets.csv")
#%%
df.head()
df.shape
df.describe()
df['Outcome'].value_counts()
df.groupby('Outcome').mean()

x = df.drop(columns = 'Outcome', axis=1)
y = df['Outcome']

#%% 
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, stratify=y, random_state=42)

classifier = svm.SVC(kernel='linear')

#training the Classifier
classifier.fit(x_train, y_train)

#%% 
# accuracy score on the training data
train_data_pred = classifier.predict(x_train)
training_data_acc = accuracy_score(train_data_pred, y_train)
print(f"Accuracy score of training data : %{training_data_acc*100}")

# accuracy score on the test data
test_data_pred = classifier.predict(x_test)
test_data_acc = accuracy_score(test_data_pred, y_test)

print(f"Accuracy score of training data : %{test_data_acc*100}")

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
  
#%%
import pickle
filename = 'diabets.sav'
pickle.dump(classifier, open(filename, 'wb'))
# loading the saved model
#loaded_model = pickle.load(open('trained_model.sav', 'rb'))
  