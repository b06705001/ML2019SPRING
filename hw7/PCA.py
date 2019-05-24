import os
import sys
import numpy as np 
from skimage.io import imread, imsave
# python PCA.py Aberdeen/   10.jpg   10_reconstruct.jpg
IMAGE_PATH = sys.argv[1]

# Images for compression & reconstruction
test_image = [sys.argv[2]] 
output_image=sys.argv[3]
# Number of principal components used
k = 5
def lidir(path):
	for f in os.listdir(IMAGE_PATH):
		if not f.startswith('.'):
			yield f


def process(M): 
    M=np.copy(M)
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M


filelist = sorted(lidir(IMAGE_PATH) )
# Record the shape of images
img_shape = imread(os.path.join(IMAGE_PATH,filelist[0])).shape 

img_data = []
for filename in filelist:
    tmp = imread(os.path.join(IMAGE_PATH,filename))  
    img_data.append(tmp.flatten())
training_data = np.array(img_data).astype('float32')

# Calculate mean & Normalize
mean = np.mean(training_data, axis = 0)  
training_data -= mean 
# Use SVD to find the eigenvectors 
u, s, v = np.linalg.svd(training_data, full_matrices = False)  


for x in test_image: 
    # Load image & Normalize
    picked_img = imread(os.path.join(IMAGE_PATH,x))  
    X = picked_img.flatten().astype('float32') 
    X -= mean
    
    # Compression
    weight = np.array([X.dot(np.transpose(v[i])) for i in range(k)])  
    
    # Reconstruction
    reconstruct = process(weight.dot(v[:k]) + mean)
    imsave(output_image, reconstruct.reshape(img_shape)) 
    """
#Report Problem 1.a
average = process(mean)
imsave('average.jpg', average.reshape(img_shape))  
#Report Problem 1.b
for x in range(5):
    eigenface = process(v[x])
    imsave(str(x) + '_eigenface.jpg', eigenface.reshape(img_shape))  
#Report Problem 1.d
for i in range(5):
    number = s[i] * 100 / sum(s)
    print(number)
    """















