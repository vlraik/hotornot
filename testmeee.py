import numpy as np
from keras.models import load_model
import cv2

model = load_model("trained.h5")

myimage = cv2.imread("myimage.jpg")
myimage = np.asarray(myimage)


model.predict(myimage)