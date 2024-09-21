import pandas as pd
import numpy as np
#import torch
#from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# from google.colab import drive

# drive.mount('/content/gdrive')
# df = pd.read_csv('/content/gdrive/MyDrive/train.csv')
df = pd.read_csv('train.csv')

#allot and delete columns of the data
X = df
X = X.drop('ACTION', axis=1) #drop target column

#X= X.drop('ROLE_CODE',axis=1) #drop role code, overlapping with role family
y = df['ACTION']
#X = pd.get_dummies(X)

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalize the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.fit_transform(X_test)

#print scaled rdata
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy*100)
