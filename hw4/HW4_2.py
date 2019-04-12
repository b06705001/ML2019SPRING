import sys
import csv
import numpy as np
import time
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, array_to_img
# util function to convert a tensor into a valid image

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
   # x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

model=load_model('my_model.h5', compile=False)
layer_dict = dict([(layer.name, layer) for layer in model.layers])

plt.figure()
fg = plt.figure()    
plt.axis('off')
plt.title("Filters of layer conv2d_1")
count=1
for i in range(32):
    filter_index = i  # can be any integer from 0 to 511, as there are 512 filters in that layer
    input_img=model.input
    # build a loss function that maximizes the activation
# of the nth filter of the layer considered
    layer_output = layer_dict['activation_1'].output
    loss = K.mean(layer_output[:, :, :, filter_index])

# compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [grads])


# we start from a gray image with some noise
    np.random.seed(16);
    input_img_data = np.random.random((1,42,42,1))

# run gradient ascent for 20 steps
    for i in range(20):
        grads_value = iterate([input_img_data])[0]
        input_img_data += grads_value * 0.1



    input_img_data= np.reshape(( input_img_data),(42,42) )
    
    img = input_img_data
   # img = deprocess_image(img)
    fg.add_subplot(4, 8, count)
    plt.imshow(img, cmap='Blues')
    plt.axis('off')
    count+=1

plt.savefig(sys.argv[2]+'fig2_1'+'.jpg')

plt.axis('off')
#       python HW4(2).py train.csv "C:\Users\peter yang\Desktop\MLHW4\"