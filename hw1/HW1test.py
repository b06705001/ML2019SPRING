import sys
import csv
import numpy as np
data=[[]for i in range(18) ]
count=0
#fp = open(sys.argv[2],w)
with open(sys.argv[1], newline='') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		del row[0:2]
		for j in range(len(row)):
			data[(count)%18].append(row[j])
		count+=1
for i in range(len(data[10])):
	if data[10][i]=="NR":
		data[10][i]=0
testx=[]
count1=0

for i in range(len(data[9])-8):
	if i%9 !=0:
		count1+=1
		continue
	testx.append([])
	for j in range(18):
		for k in range(9):	
			testx[i-count1].append(data[j][i+k])#第i次SAMPLE的第j ROW的資料的SAMPLE組合
	testx[i-count1].append(1)#BIAS
w=np.zeros([163,1],dtype=float)
w=np.load('hw1.npy')

print(w)
testx=np.array(testx,dtype=float)
print(testx[0])
print(testx.shape)
y=np.empty([testx.shape[0],1],dtype=float)
y=np.dot(testx,w)
with open(sys.argv[2], 'w', newline='') as csvfile:
  # 建立 CSV 檔寫入器
  writer = csv.writer(csvfile)
  writer.writerow(["id","value"])
  for i in range(y.shape[0]):
  	writer.writerow(["id_"+str(i),y[i][0]])


#python HW1test.py "C:\Users\peter yang\Downloads\test .csv" "C:\Users\peter yang\Desktop\ans.csv"