from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing import image
from keras import models

model_path="/home/shivam/Documents/orientation_detection/Mynewmodel_10.h5"
model = load_model(model_path)
'''
# Extracts the outputs of the top 12 layers
 # Returns a list of five Numpy arrays: one array per layer activation
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
'''
# Creates a model that will return these outputs, given the model input
layer_outputs = [layer.output for layer in model.layers[:20]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

weights_layer0 = np.zeros_like(model.layers[0].get_weights()[0])
bias_layer0 = np.zeros_like(model.layers[0].get_weights()[1])
l=[]
l.append(np.zeros_like(weights_layer0))
l.append(np.zeros_like(bias_layer0))
img_path = "-90.674.jpg"
img_p = '/home/shivam/Documents/orientation_detection/Dataset/' + img_path
result_folder="/home/shivam/Documents/orientation_detection/Dataset/Newintermediate_-90/"
img = image.load_img(img_p, target_size=(400, 400))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
activations = activation_model.predict(img_tensor)

layer_names = []
for layer in model.layers[:20]:
    layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot
images_per_row = 1

for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
    n_features = layer_activation.shape[-1]  # Number of features in the feature map
    size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
    n_cols = 1  # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for ch in range(n_features):
        channel_image = layer_activation[0,
                        :, :,
                        ch]
        channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
        channel_image /= channel_image.std()
        #print(channel_image.mean())
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        

        if (layer_name == 'dropout_1'):
            cv2.imwrite(result_folder+"layer1/" + str(ch)+img_path,
                        channel_image)
        elif (layer_name == 'dropout_2'):
            cv2.imwrite(result_folder+"layer2/" + str(ch)+img_path,
                        channel_image)
        elif (layer_name == 'dropout_3'):
            cv2.imwrite(result_folder+"layer3/" + str(ch)+img_path,
                        channel_image)
        elif (layer_name == 'dropout_4'):
            cv2.imwrite(result_folder+"layer4/" + str(ch)+img_path,
                        channel_image)
        elif (layer_name == 'dropout_5'):
            cv2.imwrite(result_folder+"layer5/" + str(ch)+img_path,
                        channel_image)

