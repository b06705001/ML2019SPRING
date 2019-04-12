import sys
import csv
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
#import cv2
from keras.preprocessing.image import img_to_array, array_to_img
typecount=7
label=[-1]*typecount
data=[]
for i in range(7):
	data.append([])
dataat=[0]*7
count=0
count3=0
ishere=[False]*typecount
with open(sys.argv[1], newline='') as csvFile:
#with open('/Users/peter yang/Downloads/train.csv', newline='') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		if count==0:
			count+=1
			continue
		label[count-1]=row[0]
		lab=int(label[count-1])
		if ishere[lab]==False:	
			for j in range(1,len(row)):		
				data[lab].append(row[j])
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


data=np.reshape(augdata,(7,1,42,42,1))
model = load_model('my_model.h5', compile=False)
for i in range(typecount):
	preds=model.predict(data[i])
	n=np.argmax(preds[0])
	inp = model.input
	outp = model.output
	grad = K.gradients(K.max(outp), inp)[0]
	a= K.function([inp],[grad])
	heatmap=a([data[i]])
	heatmap = np.abs(heatmap)
	heatmap=np.reshape(heatmap,(42,42))
	heatmap /= (np.max(heatmap))
	
	plt.matshow(heatmap)
	plt.colorbar()
	#plt.show()
	plt.savefig(sys.argv[2]+'fig1_'+ str(i)+'.jpg')
"""
	img=data[i]
	img=np.reshape(img,(42,42,1))
	img*=255
	heatmap = np.uint8(255 * heatmap)#将热力图转换为RGB格式
	for i in range(42):
		for j in range(42):
			if(heatmap[i][j]<150):
				heatmap[i][j]=0
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)# 将热力图应用于原始图像
 
	superimposed_img = heatmap * 0.4 + img# 这里的0.4是热力图强度因子
 
	cv2.imwrite('rep'+str(i)+'.jpg', superimposed_img)


"""
#		python HW4(1).py train.csv "C:\Users\peter yang\Desktop\MLHW4"