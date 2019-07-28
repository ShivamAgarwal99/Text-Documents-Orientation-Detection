#importing important libraries
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
#loading saved model
model = load_model('/home/shivam/Documents/orientation_detection/Mynewmodel_10.h5')
img_p = '/home/shivam/Documents/pdf/images/6.jpg' 
#converting image to suitabe image for inputing in model 
img = image.load_img(img_p, target_size=(400, 400))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
#predicting image
# 0 : -90 
#1 : 0
#2 : 90
output = model.predict(img_tensor)
print(np.argmax(output))
