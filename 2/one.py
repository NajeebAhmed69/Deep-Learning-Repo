import numpy as np 
import pandas as pd


df = pd.read_csv('Churn_Modelling.csv')


df.drop(columns = ['RowNumber','CustomerId','Surname'],inplace=True)
df = pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)
X = df.drop(columns=['Exited'])
y = df['Exited'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)

# Import TensorFlow
import tensorflow as tf

# Access Keras directly from TensorFlow
from tensorflow import keras

# Access Keras Layers
from tensorflow.keras import layers

# Example: Building a simple model using Keras layers
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(10, activation='softmax')
])

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
