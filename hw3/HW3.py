import tensorflow as tf
import keras
import sys
import csv
import math
import numpy as np
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
def scheduler(epoch):
    lr = 0.01
    if epoch > 80:
        frac = (epoch - 80) //5
        decay_factor = 0.9 ** frac
        K.set_value(model.optimizer.lr, lr *decay_factor)
        print("lr changed to {}".format(K.get_value(model.optimizer.lr)))
    return K.get_value(model.optimizer.lr)

typecount=7
label=[]
data=[]
count=0
with open(sys.argv[1], newline='') as csvFile:
#with open('/Users/peter yang/Downloads/train.csv', newline='') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		if count==0:
			count+=1
			continue
		data.append([])
		label.append([])
		label[count-1].append(row[0])
		for j in range(1,len(row)):		
			data[count-1].append(row[j])
		count+=1
sdata=[]
for i in range(len(data)):
	sdata.append(data[i][0].split())

sdata=np.array(sdata,dtype=float)
sdata=np.reshape(sdata,(len(data),48,48))
augmantdata=[]
auglabel=[]
crosssize=[[0,0],[0,6],[6,0],[6,6],[3,3]]
count1=0
for i in range(len(sdata)):
	for j in range(5):
		augmantdata.append([])
		auglabel.append(label[i][0])
		for x in range(crosssize[j][0],42+crosssize[j][0]):
			for y in range(crosssize[j][1],42+crosssize[j][1]):
				augmantdata[count1].append(sdata[i][x][y])
		count1+=1
trainlabel=[]
for i in range(len(augmantdata)):
	trainlabel.append([])
	for j in range(typecount):
		if(j==int(auglabel[i])):
			trainlabel[i].append(1)
		else:
			trainlabel[i].append(0)
augmantdata=np.array(augmantdata,dtype=float)
augmantdata/=255.0
del sdata
del label
del auglabel
trainlabel=np.array(trainlabel,dtype=int)
augmantdata=np.reshape(augmantdata,(5*len(data),42,42,1))

vdata=[]
vlabel=[]
count=0



"""
#################################################################################################
with open(sys.argv[2], newline='') as csvFile:
#with open('/Users/peter yang/Downloads/train.csv', newline='') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		if count==0:
			count+=1
			continue
		vdata.append([])
		vlabel.append([])
		vlabel[count-1].append(row[0])
		for j in range(1,len(row)):		
			vdata[count-1].append(row[j])
		count+=1
svdata=[]
trainvlabel=[]
leng=len(vdata)
leng=int(leng)
#############################
##########################
for i in range(leng):
	svdata.append(vdata[i][0].split())
svdata=np.array(svdata,dtype=float)
svdata=np.reshape(svdata,(leng,48,48))
count1=0
augvdata=[]
augvlabel=[]
for i in range(len(svdata)):
	for j in range(5):
		augvdata.append([])
		augvlabel.append(vlabel[i][0])
		for x in range(crosssize[j][0],42+crosssize[j][0]):
			for y in range(crosssize[j][1],42+crosssize[j][1]):
				augvdata[count1].append(svdata[i][x][y])
		count1+=1

augvdata=np.array(augvdata,dtype=float)
augvdata/=255.0
for i in range(len(augvdata)):
	trainvlabel.append([])
	for j in range(typecount):
		if(j==int(augvlabel[i][0])):
			trainvlabel[i].append(1)
		else:
			trainvlabel[i].append(0)
trainvlabel=np.array(trainvlabel,dtype=int)
augvdata=np.reshape(augvdata,(5*leng,42,42,1))
"""
drop=0.5
model = Sequential()
model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer='he_normal', input_shape=(42,42,1)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Conv2D(128, (2, 2), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization()) 
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())


model.add(Dropout(drop))


model.add(Dense(units=512,
				kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop))

model.add(Dense(units=7))
model.add(Activation('softmax'))
sgd=SGD(lr=0.01,momentum=0.9,decay=5e-4,nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])






gen = ImageDataGenerator( horizontal_flip=True )

gen.fit(augmantdata)

train_generator = gen.flow(augmantdata, trainlabel, batch_size=128)

cw={0:1.8,
	1:1,
	2:1.7,
	3:1,
	4:1.5,
	5:2.3,
	6:1.5}

"""
test_gen = ImageDataGenerator()
test_gen.fit(augvdata)
test_generator = test_gen.flow(augvdata, trainvlabel, batch_size=128)
"""
from keras.callbacks import ReduceLROnPlateau
learning_rate_function = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=1,
                                            min_delta=0.00001, 
                                            verbose=1, 
                                            factor=0.1)


lrate = LearningRateScheduler(scheduler)
callbacklist=[lrate]
model.fit_generator(train_generator,steps_per_epoch=120,
                                   epochs=250,
                                   callbacks=callbacklist,
                                    verbose=1,
                                    shuffle=True,
                                #    class_weight=cw,
                                #   validation_data=(augvdata,trainvlabel)
                                    )

#model.fit(sdata,trainlabel,batch_size=100,epochs=50)
model.save('my_model.h5')
result=model.evaluate(augmantdata,trainlabel,batch_size=1200)
print("result",result[1])
#	python HW3.py train.csv test1.csv