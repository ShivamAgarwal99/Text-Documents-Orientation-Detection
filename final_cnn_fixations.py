import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras import models

model = load_model('/home/developers/orientation/Mynewmodel_10.h5')
img_path = "0.37.jpg"
img_p = '/home/developers/orientation/Dataset/NewTrain/NewTrain/' + img_path
img = image.load_img(img_p, target_size=(400, 400))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

#returns model output after given layer_name    
def get_outputs(layer_name):
    intermediate_layer_model = models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(img_tensor)
    return intermediate_output

def max_pool_back_track(activations_prev_layer,nodes,f):
    out_list=[]
    for node in nodes:
        if f == 0:
            node_3d = node
        else:
            node_3d = np.unravel_index(node,(10,10,512))
        receptive_feild =activations_prev_layer[2*node_3d[0]:2*node_3d[0]+2,2*node_3d[1]:2*node_3d[1]+2,node_3d[2]]
        t = np.argmax(receptive_feild)
        u = np.unravel_index(t,[2,2])
        new = [u[0]+2*node_3d[0], u[1] + 2*node_3d[1]]
        out_list.append([new[0],new[1],node_3d[2]])
    return out_list

def conv_back_track(activations_prev_layer,weights,nodes,stride,store):
    conv_out_list=[]
    channel_set=set()
    for node in nodes:
        node_3d = node
        if stride*node_3d[0] == 0 :
            left = 0
        else:
            left = stride*node_3d[0]-1
            
        if stride*node_3d[1] == 0 :
            top = 0
        else:
            top = stride*node_3d[1]-1
        
        receptive_feild =activations_prev_layer[left :left +3,top:top+3,:] 
        product = np.multiply(receptive_feild ,  weights[:,:,:,node_3d[2]])
        channels = np.argwhere(np.sum(product,axis = (0,1)) > .5)
        #c = np.sum(product,axis = (0,1))1
        #channels = c.argsort()[::-1][:10]
        for ch in channels:
            conv_out_list.append([node_3d[0],node_3d[1],ch[0]])
        channel_set = channel_set.union({ch[0] for ch in channels})
    if (len(channel_set)>0) and store == 1:
           with open("/home/developers/orientation/final_layer_fixation/channel_set_-90.csv", "a") as outfile:
               for entries in channel_set:
                   outfile.write(str(entries))
                   outfile.write("\n")
        
    return conv_out_list

#dense2


weights_dense2 = model.layers[24].get_weights()[0].T
output_last = get_outputs("dense_2")
classified_value = output_last.argmax(axis =1)
active_weights = weights_dense2[classified_value]
print("Classified Value",classified_value)
output_dense2 = get_outputs('dropout_6')
product_dense2 = np.multiply(active_weights,output_dense2).T
nodes_dense2 = []
for i in range(0,len(product_dense2)):
    if product_dense2[i] > 0.1:
        nodes_dense2.append(i)

print("Dense")
   
#flatten
weights_dense1 = model.layers[21].get_weights()[0].T
output_dense1 = get_outputs('flatten_1')
product_dense1 = np.multiply(weights_dense1,output_dense1)
nodes_dense1 = []
for i in range(0,len(nodes_dense2)):
    for j in range(0,51200):
        if product_dense1[nodes_dense2[i]][j] > .1:
            nodes_dense1.append(j)     
nodes_dense1 = set(nodes_dense1)
print("Flatten")

#maxPool_5
output_maxpool_5 = get_outputs('batch_normalization_5').reshape(21,21,512)
outlist_maxpool5 = max_pool_back_track(output_maxpool_5,nodes_dense1,1)
print("maxPool_5")

#conv2d_5
output_conv2d_5 = get_outputs('dropout_4').reshape(23,23,256)
weights_conv2d_5 = model.layers[16].get_weights()[0]
outlist_conv2d_5 = conv_back_track(output_conv2d_5,weights_conv2d_5,outlist_maxpool5,1,0)
print("conv_5")
#maxPool_4
output_maxpool_4 = get_outputs('batch_normalization_4').reshape(46,46,256)
outlist_maxpool4 = max_pool_back_track(output_maxpool_4,outlist_conv2d_5,0)
print("maxPool_4")
#conv2d_4
output_conv2d_4 = get_outputs('dropout_3').reshape(48,48,128)
weights_conv2d_4 = model.layers[12].get_weights()[0]
outlist_conv2d_4 = conv_back_track(output_conv2d_4,weights_conv2d_4,outlist_maxpool4,1,0)
print("conv_4")
#maxPool_3
output_maxpool_3 = get_outputs('batch_normalization_3').reshape(96,96,128)
outlist_maxpool3 = max_pool_back_track(output_maxpool_3,outlist_conv2d_4,0)
print("maxPool_3")
#conv2d_3
output_conv2d_3 = get_outputs('dropout_2').reshape(98,98,64)
weights_conv2d_3 = model.layers[8].get_weights()[0]
outlist_conv2d_3 = conv_back_track(output_conv2d_3,weights_conv2d_3,outlist_maxpool3,1,0)
print("conv_3")
#maxPool_2
output_maxpool_2 = get_outputs('batch_normalization_2').reshape(197,197,64)
outlist_maxpool2 = max_pool_back_track(output_maxpool_2,outlist_conv2d_3,0)
print("maxPool_2")
#conv2d_2
output_conv2d_2 = get_outputs('dropout_1').reshape(199,199,32)
weights_conv2d_2 = model.layers[4].get_weights()[0]
outlist_conv2d_2 = conv_back_track(output_conv2d_2,weights_conv2d_2,outlist_maxpool2,1,0)
print("conv_2")

#maxPool_1
output_maxpool_1 = get_outputs('batch_normalization_1').reshape(398,398,32)
outlist_maxpool1 = max_pool_back_track(output_maxpool_1,outlist_conv2d_2,0)
print("maxPool_1")
#conv2d_1
output_conv2d_1 = img_tensor.reshape(400,400,3)
weights_conv2d_1 = model.layers[0].get_weights()[0]
outlist_conv2d_1 = conv_back_track(output_conv2d_1,weights_conv2d_1,outlist_maxpool1,1,0)
print("conv_1")

#storing coordinates in csv file
import csv
with open("/home/developers/orientation/final_layer_fixation/final_fixation_0.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(outlist_conv2d_1)
    
#plotting coordinates over image
  
import cv2
import pandas as pd
img = cv2.imread(img_p,cv2.IMREAD_COLOR)
img = cv2.resize(img,(400,400))
coordinates = pd.read_csv (r'/home/developers/orientation/final_layer_fixation/final_fixation_0.csv', header=None)
a =coordinates[0]
b =coordinates[1]

for x,y in zip(a,b):
    cv2.circle(img,(y,x), 1, (0,0,255), -1)
    cv2.imwrite( "/home/developers/orientation/final_layer_fixation/test1.jpg", img )

