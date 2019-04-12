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
trainlabel=[]

for i in range(len(data)):
	trainlabel.append([])
	for j in range(typecount):
		if(j==int(label[i][0])):
			trainlabel[i].append(1)
		else:
			trainlabel[i].append(0)
trainlabel=np.array(trainlabel,dtype=int)

cut=20
leng=len(data)/cut
leng=int(leng)
output=[]
crosssize=[[0,0],[0,6],[6,0],[6,6],[3,3]]
model = keras.models.load_model('my_model.h5')
for c in range(cut-1):
	sdata=[]
	for i in range(c*leng,(c+1)*leng):
		sdata.append(data[i][0].split())
	sdata=np.array(sdata,dtype=float)
	sdata=np.reshape(sdata,(len(sdata),48,48))
	augmantdata=[]
	count1=0
	for i in range(len(sdata)):
		for j in range(5):
			augmantdata.append([])
			for x in range(crosssize[j][0],42+crosssize[j][0]):
				for y in range(crosssize[j][1],42+crosssize[j][1]):
					augmantdata[count1].append(sdata[i][x][y])
			count1+=1
	augmantdata=np.array(augmantdata,dtype=float)
	augmantdata/=255.0
	augmantdata=np.reshape(augmantdata,(5*len(sdata),42,42,1))
	ans1=model.predict(augmantdata,batch_size=5*len(sdata))
	print(ans1)
	print(ans1.shape)
	del sdata
	del augmantdata
	for i in range(0,len(ans1),5):
		max=0
		maxnum=0
		sum=[0,0,0,0,0,0,0]

		for j in range(len(ans1[0])):
			for k in range(5):
				sum[j]+=ans1[i+k][j]
			sum[j]/=5
			if(sum[j]>max):
				max=sum[j]
				maxnum=j
		output.append(maxnum)
sdata=[]
for i in range((cut-1)*leng,len(data)):
	sdata.append(data[i][0].split())
sdata=np.array(sdata,dtype=float)
sdata=np.reshape(sdata,(len(sdata),48,48))
augmantdata=[]
crosssize=[[0,0],[0,6],[6,0],[6,6],[3,3]]
count1=0
for i in range(len(sdata)):
	for j in range(5):
		augmantdata.append([])
		for x in range(crosssize[j][0],42+crosssize[j][0]):
			for y in range(crosssize[j][1],42+crosssize[j][1]):
				augmantdata[count1].append(sdata[i][x][y])
		count1+=1
augmantdata=np.array(augmantdata,dtype=float)
augmantdata/=255.0
augmantdata=np.reshape(augmantdata,(5*len(sdata),42,42,1))
model = keras.models.load_model('my_model.h5')
ans1=model.predict(augmantdata,batch_size=5*len(sdata))
print(ans1)
print(ans1.shape)
del sdata
del augmantdata
for i in range(0,len(ans1),5):
	max=0
	maxnum=0
	sum=[0,0,0,0,0,0,0]
	for j in range(len(ans1[0])):
		for k in range(5):
			sum[j]+=ans1[i+k][j]
		sum[j]/=5
		if(sum[j]>max):
			max=sum[j]
			maxnum=j
	output.append(maxnum)





correct=0
"""
n=np.zeros([7,7],dtype=int)
for i in range(len(output)):
	if output[i]==int(label[i][0]):
		correct+=1
		n[output[i]][output[i]]+=1
	else:
		n[output[i]][int(label[i][0])]+=1
correct/=len(output)
print(correct)
print(n)
"""
with open(sys.argv[2], 'w', newline='') as csvfile:
#   建立 CSV 檔寫入器
	writer = csv.writer(csvfile)
	writer.writerow(["id","label"])
	for i in range(len(output)):
		writer.writerow([str(i),output[i]])
#		python HW3_test.py test.csv ans.csv