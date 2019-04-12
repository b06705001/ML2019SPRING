import sys
import csv
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
from numpy.lib.arraypad import _as_pairs
from skimage.segmentation import slic
from keras.preprocessing.image import img_to_array, array_to_img
from lime import lime_image
import skimage.color as color

explainer = lime_image.LimeImageExplainer()

def explain(instance, predict_fn,segmentation_fn, **kwargs):
  np.random.seed(16)
  return explainer.explain_instance(image=instance,classifier_fn=predict_fn,segmentation_fn=segmentation_fn,num_samples=1000 ,**kwargs)
model=load_model('my_model.h5')
def predict(input):
	
    # Returns a predict function which returns the probabilities of labels ((7,) numpy array)
    
    temp = input.transpose((3, 0, 1, 2))
    temp = np.reshape(temp[0], ( 10,42, 42, 1))  
    p=model.predict(temp)
    return p
    # ex: return model(data).numpy()
    # TODO:
    # return ?

def segmentation(input):
    # Input: image numpy array
   # input=color.rgb2gray(input) 
    # Returns a segmentation function which returns the segmentation labels array ((48,48) numpy array)
    a=slic(input,n_segments=100,sigma=0.5, compactness=10)
    return a

    # TODO:
    # return ?
typecount=7
label=[-1]*typecount
data=[]
for i in range(7):
	data.append([])
dataat=[0]*7
count=0
count3=0
ishere=[False]*typecount
l=[]*7
with open(sys.argv[1], newline='') as csvFile:
#with open('/Users/peter yang/Downloads/train.csv', newline='') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		if count==0:
			count+=1
			continue
		lab=int(label[count-1])
		if ishere[lab]==False:	
			for j in range(1,len(row)):		
				data[lab].append(row[j])

			label[count-1]=row[0]
			ishere[lab]=True
			count+=1
			dataat[lab]=count3
		if count==8:
			break
		count3+=1
sdata=[]
count1=0
for i in range(len(data)):
	sdata.append(data[i][0].split())
data=np.array(sdata,dtype=float)
data/=255
data=np.reshape(data,(7,48,48))
crosssize=[[3,3]]
augdata=[]
for i in range(len(data)):
	augdata.append([])
	for x in range(crosssize[0][0],42+crosssize[0][0]):
		for y in range(crosssize[0][1],42+crosssize[0][1]):
			augdata[count1].append(data[i][x][y])
	count1+=1

data=np.reshape(augdata,(7,42*42,1))
x_train_rgb= np.concatenate((data, data, data), axis=-1)

x_train_rgb=np.reshape(x_train_rgb,(7,42,42,3))
for i in range(7):

	print(label[i])
	explaination =explain( x_train_rgb[i], 
                            predict,
                            segmentation
                        )

	image, mask = explaination.get_image_and_mask(
                                label=i,
                                positive_only=False,
                                hide_rest=False,
                                num_features=5,
                                min_weight=0.0
                            )

# save the image

	image /= np.max(image)
	plt.imsave('fig3_' + str(i) + '.jpg', image)
#	python HW4(3).py train.csv