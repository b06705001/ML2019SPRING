import sys
import csv
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, array_to_img

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
    x = x.transpose((0,1,4,2, 3))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
typecount=7
label=[]
data=[]
dataat=[]
count=0
count3=0
with open(sys.argv[1], newline='') as csvFile:
#with open('/Users/peter yang/Downloads/train.csv', newline='') as csvFile:
    rows = csv.reader(csvFile, delimiter=',')
    for row in rows:
        if count==0:
            count+=1
            continue
        data.append([])
        for j in range(1,len(row)):     
            data[count-1].append(row[j])      
        count+=1
        break
sdata=[]
count1=0
for i in range(len(data)):
    sdata.append(data[i][0].split())
data=np.array(sdata,dtype=float)
data/=255
data=np.reshape(data,(1,48,48))
crosssize=[[3,3]]
augdata=[]
for i in range(len(data)):
    augdata.append([])
    for x in range(crosssize[0][0],42+crosssize[0][0]):
        for y in range(crosssize[0][1],42+crosssize[0][1]):
            augdata[count1].append(data[i][x][y])
    count1+=1


plt.figure()
fg = plt.figure()    
plt.axis('off')
plt.title("output of layer conv2d_1(image 1)")
count=1
data=np.reshape(augdata,(1,42,42,1))
model=load_model('my_model.h5', compile=False)
layer_dict = dict([(layer.name, layer) for layer in model.layers])

input_img=model.input
layer_output = layer_dict['conv2d_1'].output
iterate = K.function([input_img], [layer_output])
output=iterate([data])
print(output)
output=np.array(output)
print(output.shape)
output=deprocess_image(output)
output=np.reshape(output,(64,42,42))
count=1
for i in range(32):
	fg.add_subplot(4, 8, count)
	plt.imshow(output[i], cmap='Blues')
	plt.axis('off')
	count+=1


plt.savefig(sys.argv[2]+'fig2_2'+'.jpg')

plt.axis('off')
#       python HW4(2_1).py train.csv "C:\Users\peter yang\Desktop\MLHW4/"
