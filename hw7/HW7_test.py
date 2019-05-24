import numpy as np
np.random.seed(1337)  # for reproducibility
 
from keras.models import Model #泛型模型
from keras.layers import Dense, Input
import keras.backend as K
import os
import sys
from sklearn.cluster import KMeans
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, load_model
from PIL import Image
import csv
import sklearn

x_train=[]
def read_directory(directory_name):
    count=0
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in sorted(os.listdir(directory_name)):
        img = Image.open(r""+directory_name  + filename)
        global array_of_img
        img=np.array(img)
        x_train.append(img)
        count+=1

model=load_model('my_model.h5')
read_directory(sys.argv[1])
get_encoded = K.function([model.layers[0].input], [model.layers[2].output])
get_decoded= K.function([model.layers[3].input], [model.layers[6].output])
X_sample=np.array(x_train,dtype=float)
X_sample/=255.
# 获取样本的 encoded 结果

X_decoded=np.empty((len(X_sample), 32, 32, 3), dtype='float32')
X_encoded = np.empty((len(X_sample), 16, 16, 8), dtype='float32')

step = 100
for i in range(0, len(X_sample), step):
    x_batch = get_encoded([X_sample[i:i+step]])[0]
    X_encoded[i:i+step] = x_batch


X_encoded_reshape = X_encoded.reshape(X_encoded.shape[0],X_encoded.shape[1]*X_encoded.shape[2]*X_encoded.shape[3])
from sklearn.decomposition import PCA


pca = PCA(n_components=300,whiten=True)  
"""
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_encoded_reshape)
cov_mat = np.cov(X_train_std.T)                              #cov_mat 共變異數矩陣
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)    #eigen_vals為特徵值，eigen_vecs為 特徵向量
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
"""
x_train_pca= pca.fit_transform(X_encoded_reshape)
"""
X=pca.inverse_transform(x_train_pca)
X=np.reshape(X,(len(X),16,16,8))
for i in range(0, len(X), step):
    x_batch = get_decoded([X[i:i+step]])[0]
    X_decoded[i:i+step] = x_batch
print(X_decoded[0])
print(X_decoded[0].shape)

X_decoded*=255.
X_decoded=np.uint8(X_decoded)

#############################################################################33

plt.figure()
fg = plt.figure()    
plt.axis('off')
plt.title('Original Image')
for i in range(1, 33):

    fg.add_subplot(4,8,i)
    plt.imshow(X_sample[i])
    plt.axis('off')

plt.savefig('fig_O'+'.jpg')
plt.show()

plt.figure()
fig = plt.figure()    
plt.axis('off')
plt.title("reconstruct image")
for i in range(1, 33):

    fig.add_subplot(4,8,i)
    plt.imshow(X_decoded[i])
    plt.axis('off')

plt.savefig('fig_R'+'.jpg')
plt.show()
"""

km = KMeans(n_clusters=2, random_state=0)
km.fit(x_train_pca)
cluster_labels = km.labels_


count=0
data=[]
with open(sys.argv[2], newline='') as csvFile:
#with open('/Users/peter yang/Downloads/train.csv', newline='') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		if count==0:
			count+=1
			continue
		data.append([])
		data[count-1].append(row[1])
		data[count-1].append(row[2])
	#	data.append(emoji.demojize(row[1]))
		count+=1
ans=[]
data=np.array(data,dtype=int)
for i in range(len(data)):
	if(cluster_labels[data[i][0]-1]==cluster_labels[data[i][1]-1]):
		ans.append(1)
	else:
		ans.append(0)


with open(sys.argv[3], 'w', newline='') as csvfile:
#   建立 CSV 檔寫入器
	writer = csv.writer(csvfile)
	writer.writerow(["id","label"])
	for i in range(len(ans)):
		writer.writerow([str(i),ans[i]])



"""
for n_image in range(0, 5):
    
    plt.figure(figsize=(12,4))

    plt.subplot(1,4,1)
    plt.imshow(X_sample[n_image][:,:,::-1])
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1,4,2)
    plt.imshow(encoded_sample[n_image].mean(axis=-1))
    plt.axis('off')
    plt.title('Encoded Mean')

    plt.subplot(1,4,3)
    plt.imshow(encoded_sample[n_image].max(axis=-1))
    plt.axis('off')
    plt.title('Encoded Max')

    plt.subplot(1,4,4)
    plt.imshow(encoded_sample[n_image].std(axis=-1))
    plt.axis('off')
    plt.title('Encoded Std')

    plt.show()
"""

#			python HW7_test.py images/ test_case.csv ans.csv