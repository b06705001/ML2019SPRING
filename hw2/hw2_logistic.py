import sys
import csv
import math
import numpy as np
from numpy.linalg import inv
num=np.load('choose.npy')
w=np.load('my_array.npy')
w2=np.load('w2.npy')
test_x=[]
count1=0
m=0
count=0
with open(sys.argv[1], newline='') as csvFile:
#with open('/Users/peter yang/Downloads/train.csv', newline='') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		if count==0:
			count+=1
			continue
		test_x.append([])
		for j in range(len(row)):
			test_x[count-1].append(row[j])
		count+=1
seq=[0,1,3,4,5, 19, 25, 29, 35, 45, 55, 56]
for i in range(len(test_x)):
	for j in range(len(seq)):
		test_x[i].append(int(test_x[i][seq[j]])**2)
	test_x[i].append(1)
test_x=np.array(test_x,dtype=float)
data2=[]
for i in range(len(test_x)):
	data2.append([])
	for j in range(len(num)):
		data2[i].append(test_x[i][num[j]])
	data2[i].append(1)

data2=np.array(data2,dtype=float)
y1=np.zeros([len(data2),1],dtype=float)


v=np.zeros([1,len(test_x[0])],dtype=float)
u=np.zeros([1,len(test_x[0])],dtype=float)


for i in range(len(test_x)):
	for j in range(len(test_x[0])-1):
		u[0][j]=u[0][j]+test_x[i][j]
u/=len(test_x)
for i in range(len(test_x)):
	for j in range(len(test_x[0])-1):
		v[0][j]=v[0][j]+(test_x[i][j]-u[0][j])**2
v=v/len(test_x)
v=np.sqrt(v)
for i in range(len(test_x)):
	for j in range(len(test_x[0])-1):
		if v[0][j]==0:
			continue
		test_x[i][j]=test_x[i][j]-u[0][j]
		test_x[i][j]=test_x[i][j]/v[0][j]

v1=np.zeros([1,len(data2[0])],dtype=float)
u1=np.zeros([1,len(data2[0])],dtype=float)


for i in range(len(data2)):
	for j in range(len(data2[0])-1):
		u1[0][j]=u1[0][j]+data2[i][j]
u1/=len(data2)
for i in range(len(data2)):
	for j in range(len(data2[0])-1):
		v1[0][j]=v1[0][j]+(data2[i][j]-u1[0][j])**2
v1=v1/len(data2)
v1=np.sqrt(v1)
for i in range(len(data2)):
	for j in range(len(data2[0])-1):
		if v1[0][j]==0:
			continue
		data2[i][j]=data2[i][j]-u1[0][j]
		data2[i][j]=data2[i][j]/v1[0][j]
y1=1/(1+np.exp(-data2.dot(w)))
y2=np.zeros([len(test_x),1],dtype=float)
y2=1/(1+np.exp(-test_x.dot(w2)))


print(y1)
with open(sys.argv[2], 'w', newline='') as csvfile:
#   建立 CSV 檔寫入器
	writer = csv.writer(csvfile)
	writer.writerow(["id","label"])
	for i in range(len(y1)):
		if (y1[i]-0.5)<0.15:
			y1[i]=y2[i]
		if y1[i]<0.5:
			writer.writerow([str(i+1),0])
		else:
			writer.writerow([str(i+1),1])


#python HW2_test.py X_test ans.csv