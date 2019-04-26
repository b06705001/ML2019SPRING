import os
from numpy.random import seed
seed(8)    
from tensorflow import set_random_seed
set_random_seed(1337)    
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from PIL import Image
import keras
import sys
import csv
import math
import numpy as np
import gc
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D  ,Activation
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.initializers import he_normal
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import preprocess_input ,decode_predictions
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras_applications.resnet import ResNet101
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        if(filename=='.DS_Store'):
        	continue
        img = Image.open(r""+directory_name + "/" + filename)
        #img_rgb2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        global array_of_img
        img=np.array(img)
        array_of_img.append(img)
        #print(img)
        #print(array_of_img)
      #  print(array_of_img.shape)



label=np.load('label.npy')
read_directory(sys.argv[1])
array_of_img=np.array(array_of_img)
#sys.argv[1]


array_of_img=np.reshape(array_of_img,(200,224,224,3))

array_of_img=preprocess_input(array_of_img)

#model1=VGG16()
#model1=ResNet50()
#model1=DenseNet169()
model1=ResNet101(backend=keras.backend,layers = keras.layers, models = keras.models, utils = keras.utils )

# Grab a reference to the first and last layer of the neural net
model_input_layer = model1.layers[0].input
model_output_layer = model1.layers[-1].output




preds=model1.predict(array_of_img)

results = decode_predictions(preds, top=3)
for i in range(len(results)):
	results[i][0]=list(results[i][0])
output_path=sys.argv[2]

################################################################
hackimg=np.copy(array_of_img)
for n in range(200):
	img=hackimg[n]
	img=np.expand_dims(img,axis=0)
	object_type_to_fake=2*n
#	object_type_to_fake=label[n]

# Define the cost function.

		
# Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
	if n!=label[n]:
		cost_function_true = -K.log(model_output_layer[0, label[n]])
		cost_function_false=-K.log(model_output_layer[0, object_type_to_fake])
	else:
		cost_function_true = -K.log(model_output_layer[0, label[n]])
		cost_function_false=-K.log(model_output_layer[0, abs(label[n]-1)])
#	max_change_above = array_of_img[n] + 15
#	max_change_below = array_of_img[n] - 15
 
# We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
# In this case, referring to "model_input_layer" will give us back image we are hacking.
	gradient_function = K.gradients((-cost_function_true)+(cost_function_false), model_input_layer)[0]

	# Create a Keras function that we can call to calculate the current cost and gradient
	grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function_false, gradient_function])

	# In a loop, keep adjusting the hacked image slightly so that it tricks the model more and more
	# until it gets to at least 80% confidence
#	while cost < 0.99 and epoch<500:
	# Check how close the image is to our target class and grab the gradients we
	# can use to push it one more step in that direction.
	# Note: It's really important to pass in '0' for the Keras learning mode here!
	# Keras layers behave differently in prediction vs. train modes!
	cost, gradients = grab_cost_and_gradients_from_model([img, 0])
    # Move the hacked image one step further towards fooling the model
	#	if(cost>-0.1 and cost<0):
	#		gradients-=gradient_temp
	img -= np.sign(gradients) *15
	#	epoch+=1
    # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image
	img = np.clip(img, [-103.939, -116.779, -123.68], [255.0-103.939, 255.0-116.779,255.0- 123.68])
	#	img = np.clip(img, [-0.485/0.229, -0.456/0.224, -0.406/0.225] , [(1-0.485)/0.229, (1-0.456)/0.224, (1-0.406)/0.225])
	#	img = np.clip(img, max_change_below, max_change_above)
 
	#	print("Model's predicted likelihood that the image is a toaster: {:.8}%".format(cost * 100))
	hackimg[n]=img
	img=np.reshape(img,(224,224,3))
	"""
	img*=[0.229, 0.224, 0.225]
	img+=[0.485, 0.456, 0.406] 
	img*=255
	"""
	img+=[103.939, 116.779, 123.68]
	img=np.uint8(img)
	im = Image.fromarray(img)
	b, g, r = im.split()
	im = Image.merge("RGB", (r, g, b))
	if(n<10):
		front="00"
	elif (n<100):
		front="0"
	else:
		front=""
	im.save(output_path+"/"+ front+str(n)+".png")
	
	del gradient_function
	del cost_function_true
	del cost_function_false
	del grab_cost_and_gradients_from_model
	del img
	gc.collect()
	im.close()
	if n%10==0:
		K.clear_session()
		model1=ResNet101(backend=keras.backend,layers = keras.layers, models = keras.models, utils = keras.utils )
		model_input_layer = model1.layers[0].input
		model_output_layer = model1.layers[-1].output
	# De-scale the image's pixels from [-1, 1] back to the [0, 255] range

# Save the hacked image!
for i in range(len(results)):
	results[i][0]=list(results[i][0])
#print(results)

preds=model1.predict(hackimg)
results = decode_predictions(preds, top=1)

#	python HW5_FGSM.py "C:\Users\peter yang\Desktop\MLHW5\images" "C:\Users\peter yang\Desktop\MLHW5"