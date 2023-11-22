from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
import glob
import numpy as np
import pandas as pd
import cv2





resnet_model=ResNet50(weights='imagenet')

path = glob.glob('imagenesTest/*.jpg')
path


for file in path:
  #from tensorflow.keras.preprocessing import image
  img_file=cv2.imread(file)
  img_file=cv2.resize(img_file,(224,224))

  x_file=image.img_to_array(img_file)
  #print(type(x_file))
  y_file=np.expand_dims(x_file,axis=0)
  #print(type(y_file))
  z_file=preprocess_input(y_file)
  #print(type(z_file))
  preds_file=resnet_model.predict(z_file)
  #print(type(preds_file))
  #print(preds_file)
  Prediction_file=decode_predictions(preds_file, top=3)[0]
  #print(type(Prediction_file))
  print('Predicted:', Prediction_file)
