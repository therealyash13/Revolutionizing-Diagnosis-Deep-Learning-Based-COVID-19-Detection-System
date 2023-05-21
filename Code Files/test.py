import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import * 
from keras.preprocessing import image
import warnings
warnings.filterwarnings('ignore')

GREEN = '\033[92m'
RED = '\033[91m'
model = load_model('model_adv.h5')
img = image.load_img("test.jpeg", target_size = (224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
p = model.predict_classes(img)
print(p)
print('Radiologist: COVID-19 -ve')

if p==0:
    print('CNN Model: COVID-19 +ve')
else:
    print("CNN Model: COVID-19 -ve")
