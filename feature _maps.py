#this code shows 32 feauture_maps of corresponding convolutional layer 
#with unique layer_index
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
model = load_model('/home/shivam/Documents/orientation_detection/Mynewmodel_10.h5')  
layer_index = 0 #for first convolutional layer      
x1w = model.layers[layer_index].get_weights()[0][:,:,0,:]
for i in range(1,33):
    plt.subplot(6,6,i)
    plt.imshow(x1w[:,:,i],interpolation="nearest",cmap="gray")
print(x1w)
plt.show()




