#this code is giving that numbered feauture_maps which shows different types of realtion
#whether positive, negative or zero  correlation in predicting corresponding classes
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras import models
from copy import deepcopy

model = load_model('/home/shivam/Documents/orientation_detection/Mynewmodel_10.h5')
img_path = "-90.674.jpg"
img_p = '/home/shivam/Documents/orientation_detection/Dataset/' + img_path
img = image.load_img(img_p, target_size=(400, 400))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

#true_label for input image
classified_value = 0

layer = 16
nchannels=512
nchannel_prev_layer= 256


def get_outputs(layer_name):
    intermediate_layer_model = models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(img_tensor)
    return intermediate_output


def max_pool_back_track(activations_prev_layer, nodes, f):
    out_list = []
    for node in nodes:
        if f == 0:
            node_3d = node
        else:
            node_3d = np.unravel_index(node, (10, 10, 512))
        receptive_feild = activations_prev_layer[2 * node_3d[0]:2 * node_3d[0] + 2, 2 * node_3d[1]:2 * node_3d[1] + 2,
                          node_3d[2]]
        t = np.argmax(receptive_feild)
        u = np.unravel_index(t, [2, 2])
        new = [u[0] + 2 * node_3d[0], u[1] + 2 * node_3d[1]]
        out_list.append([new[0], new[1], node_3d[2]])
    return out_list


def conv_back_track(activations_prev_layer, weights, nodes, stride):
    conv_out_list = []
    for node in nodes:
        node_3d = node
        if stride * node_3d[0] == 0:
            left = 0
        else:
            left = stride * node_3d[0] - 1

        if stride * node_3d[1] == 0:
            top = 0
        else:
            top = stride * node_3d[1] - 1

        receptive_feild = activations_prev_layer[left:left + 3, top:top + 3, :]
        product = np.multiply(receptive_feild, np.squeeze(weights[:, :, :, node_3d[2]] ))
        # 3 X 3 X 256
        #
        c = np.sum(product, axis=(0, 1))
        #channels = c.argsort()[::-1][:100]
        #print(len(channels))
        channels = np.argwhere(np.sum(product,axis = (0,1)) > 0.5)
        for ch in channels:
            conv_out_list.append([node_3d[0], node_3d[1], ch])

    return conv_out_list


# dense2
#-90 0
#0 1
#90 2



#weights_dense2 = model.layers[24].get_weights()[0].T
weights_layer0 = model.layers[layer].get_weights()[0]
bias_layer0 = model.layers[layer].get_weights()[1]

print(weights_layer0.shape)
print(bias_layer0.shape)
#print(weights_layer0)
# weights_layer0[:,:,:,0] =np.zeros([3,3,3])
# weights_layer0[:,:,:,1] =np.zeros([3,3,3])
# weights_layer0[:,:,:,2] =np.zeros([3,3,3]) #~[0.02641994 0.9705949  0.00298515]
# weights_layer0[:,:,:,3] =np.zeros([3,3,3])
# weights_layer0[:,:,:,4] =np.zeros([3,3,3])
# weights_layer0[:,:,:,5] =np.zeros([3,3,3]) #[5.6193020e-08 1.0000000e+00 1.3199846e-08]
# weights_layer0[:,:,:,6] =np.zeros([3,3,3])
# weights_layer0[:,:,:,7] =np.zeros([3,3,3])
# weights_layer0[:,:,:,8] =np.zeros([3,3,3])
# weights_layer0[:,:,:,9] =np.zeros([3,3,3])
# weights_layer0[:,:,:,10] =np.zeros([3,3,3])#~[1.3951509e-05 9.9998200e-01 4.0724913e-06]
# weights_layer0[:,:,:,11] =np.zeros([3,3,3])
# weights_layer0[:,:,:,12] =np.zeros([3,3,3])
# weights_layer0[:,:,:,13] =np.zeros([3,3,3]) #~[1.3911998e-05 9.9998319e-01 2.8240415e-06]
# weights_layer0[:,:,:,14] =np.zeros([3,3,3])
# weights_layer0[:,:,:,15] =np.zeros([3,3,3])
# weights_layer0[:,:,:,16] =np.zeros([3,3,3])
# weights_layer0[:,:,:,17] =np.zeros([3,3,3])
# weights_layer0[:,:,:,18] =np.zeros([3,3,3]) #[2.0326682e-10 1.0000000e+00 9.7161786e-11]
# weights_layer0[:,:,:,19] =np.zeros([3,3,3])  #[6.210670e-12 1.000000e+00 3.950559e-12]
# weights_layer0[:,:,:,20] =np.zeros([3,3,3]) # ~8.343772e-11 1.000000e+00 4.353839e-11]
# weights_layer0[:,:,:,21] =np.zeros([3,3,3]) # [[2.8747484e-08 1.0000000e+00 7.8537905e-09]]
# weights_layer0[:,:,:,22] =np.zeros([3,3,3]) #~ [6.6648732e-07 9.9999917e-01 1.4025801e-07]
# weights_layer0[:,:,:,23] =np.zeros([3,3,3])
# weights_layer0[:,:,:,24] =np.zeros([3,3,3])
# weights_layer0[:,:,:,25] =np.zeros([3,3,3])
# weights_layer0[:,:,:,26] =np.zeros([3,3,3])
# weights_layer0[:,:,:,27] =np.zeros([3,3,3])
# weights_layer0[:,:,:,28] =np.zeros([3,3,3]) #~[[0.00501441 0.99356383 0.00142177]]
# weights_layer0[:,:,:,29] =np.zeros([3,3,3])
# weights_layer0[:,:,:,30] =np.zeros([3,3,3])  #~[[3.9316728e-03 9.9548811e-01 5.8016786e-04]]
# weights_layer0[:,:,:,31] =np.zeros([3,3,3])


out = get_outputs("dense_2")[0]
print(out)
normal_score = out[classified_value]/(out[0]+out[1]+out[2])
print(normal_score)


channel_importance = dict()
for i in range(nchannels):
    l=[]
    new_weights = deepcopy(weights_layer0)
    new_weights[:, :, :, i] = np.zeros([3, 3, nchannel_prev_layer])
    l.append(new_weights)
    l.append(bias_layer0)
    model.layers[layer].set_weights(l)

    out = get_outputs("dense_2")[0]
    score = out[classified_value]/(out[0]+out[1]+out[2])
    channel_importance[i] = normal_score - score

channel_importance_np = np.asarray([channel_importance[key] for key in range(nchannels)])

print(channel_importance_np)
sorted_indices = np.argsort(np.absolute(channel_importance_np))

#print(sorted_indices)
positive_correlation = np.squeeze(np.argwhere(channel_importance_np>0))
#print(positive_correlation)
positive_correlation =[i for i in sorted_indices if i in positive_correlation.tolist()][::-1]
print(positive_correlation)

negative_correlation =[i for i in sorted_indices if i in np.squeeze(np.argwhere(channel_importance_np<0)).tolist()][::-1]
print(negative_correlation)

zero_correlation =[i for i in sorted_indices if i in np.squeeze(np.argwhere(channel_importance_np==0)).tolist()][::-1]
print(zero_correlation)



#new_weights = deepcopy(weights_layer0)

# for ch in positive_correlation:
#     new_weights[:, :, :, ch] = np.zeros([3, 3, 3])
#
# l=[]
# l.append(new_weights)
# l.append(bias_layer0)
# model.layers[0].set_weights(l)
#
# out = get_outputs("dense_2")[0]
# print(out)
# score = out[1]/(out[0]+out[1]+out[2])
# print("Boosted score",score)
