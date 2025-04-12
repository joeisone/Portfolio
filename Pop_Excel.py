import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image


new_model=keras.models.load_model(os.path.join('MODEL','DocClassifier.h5'))

img=cv2.imread(r"C:\Users\joseph.boateng\Downloads\OIP.jpg")
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()



# invoice dataset needs better data
# revise
# and retrain model



#resize image for model
resize=tf.image.resize(img,(256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


np.expand_dims(resize,0)
yhat=new_model.predict(np.expand_dims(resize/255,0))
print(yhat)


MaxIndex=np.argmax(yhat)
print(MaxIndex)
