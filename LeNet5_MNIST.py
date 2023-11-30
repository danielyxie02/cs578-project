import numpy as np 
import matplotlib.pyplot as plt

import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D 

(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

print("Dimensions of X_train are:", X_train.shape)
print("Dimensions of y_train are:", y_train.shape)
print("Dimensions of X_valid are:", X_valid.shape)
print("Dimensions of y_valid are:", y_valid.shape)

plt.figure(figsize=(2, 4))
for k in range(4):
  plt.subplot(1, 4, k+1)
  plt.imshow(X_train[k], cmap="Greys")
  plt.axis("off")

plt.tight_layout()
plt.show()


print(y_train[0:4])

#preprocess the data
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32")
X_valid = X_valid.reshape(-1, 28, 28, 1).astype("float32")

X_train.shape

#normalize the data 
X_train /= 255
X_valid /= 255

#one hot encoder 
n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_valid = to_categorical(y_valid, n_classes)

#neural network model 

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())                                                               
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))  

model.summary()

#choosing optimizer
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

#fit the model to our data 
model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_valid, y_valid))

#Evaluate model 
model.evaluate(X_valid, y_valid)