import sys
import csv
import math
import numpy as np
from numpy.linalg import inv
train_x=[]
train_y=[]
count1=0
m=0
count=0
with open(sys.argv[5], newline='') as csvFile:
#with open('/Users/peter yang/Downloads/train.csv', newline='') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		if count==0:
			count+=1
			continue
		train_x.append([])
		for j in range(len(row)):
			train_x[count-1].append(row[j])
		count+=1
cov=np.load('cov.npy')
u0=np.load('u0.npy')
u1=np.load('u1.npy')
c0num=np.load('c0num.npy')
c1num=np.load('c1num.npy')

pc0=c0num/(c0num+c1num)
pc1=c1num/(c0num+c1num)
train_x=np.array(train_x,dtype=float)
ans=[0]*len(train_x)

pxc0=1/math.pi**(len(train_x[0])/2)*(1/np.linalg.det(cov)**1/2)
pxc1=1/math.pi**(len(train_x[0])/2)*(1/np.linalg.det(cov)**1/2)
for i in range(len(train_x)):
	e0=-(1/2)*np.dot((train_x[i]-u0),np.linalg.inv(cov)).dot(np.transpose(train_x[i]-u0))
	e1=-(1/2)*np.dot((train_x[i]-u1),np.linalg.inv(cov)).dot(np.transpose(train_x[i]-u1))
	p=pc1*pxc1*math.exp(e1-e0)/(pc1*pxc1*math.exp(e1-e0)+pc0*pxc0)
	ans[i]=p

with open(sys.argv[6], 'w', newline='') as csvfile:
#   建立 CSV 檔寫入器
	writer = csv.writer(csvfile)
	writer.writerow(["id","label"])
	for i in range(len(ans)):
		if ans[i]<=0.5:
			writer.writerow([str(i+1),0])
		else:
			writer.writerow([str(i+1),1])

#python hw2_gaussion.py train.csv test.csv X_train Y_train X_test ans.csv