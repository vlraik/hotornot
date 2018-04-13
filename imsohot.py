from keras.applications import ResNet50
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

resnet = ResNet50(include_top=False,pooling='avg',input_shape=(350,350,3))

model = Sequential()

model.add(resnet)
model.add(Dense(1))
model.layers[0].trainable = False

print(model.summary())

import numpy as np
import cv2

with open('meta.txt') as f:
    lines = [lines.split() for lines in f]

print(lines)

X = []
y = []
for i in range(len(lines)):
    im = cv2.imread("Images/"+lines[i][0])
    X.append(im)
    y.append(float(lines[i][1]))
X = np.asarray(X)
print(X.shape)
print(len(y))

model.compile(loss='mean_squared_error', optimizer=Adam())
model.fit(batch_size=32, x=X, y=y, epochs=30, validation_split=0.4)

model.layers[0].trainable = True
model.compile(loss='mean_squared_error', optimizer=Adam())
model.fit(batch_size=32, x=X, y=y, epochs=30, validation_split=0.4)

model.save("trained.h5")

myimage = cv2.imread("myimage.jpg")
myimage = np.asarray(myimage)

model.predict(myimage)