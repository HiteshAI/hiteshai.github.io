---
layout: post
title:  "Face recognition"
date:   2020-01-18 14:55:04 +0545
categories: jekyll blog

---



# **Face Recognition using Eigenfaces**
---




####We will be using [FEI Face Database](https://fei.edu.br/~cet/facedatabase.html)  for our face recognition purpose. 
*Datasets consist of faces of 200 people and each person has two frontal images (one with a neutral expression and the other with a smiling facial expression), there are 400 full frontal face images manually registered and cropped.*
<br><br>**We will use normalized, equalized and cropped frontal face images.**


Add dataset in your Colab using:


```
!wget <link> 
```




```python
! wget https://fei.edu.br/~cet/frontalimages_manuallyaligned_part1.zip
! wget https://fei.edu.br/~cet/frontalimages_manuallyaligned_part2.zip
```

    --2019-12-12 05:17:05--  https://fei.edu.br/~cet/frontalimages_manuallyaligned_part1.zip
    Resolving fei.edu.br (fei.edu.br)... 200.232.90.210
    Connecting to fei.edu.br (fei.edu.br)|200.232.90.210|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 6209322 (5.9M) [application/x-zip-compressed]
    Saving to: ‘frontalimages_manuallyaligned_part1.zip’
    
    frontalimages_manua 100%[===================>]   5.92M  3.23MB/s    in 1.8s    
    
    2019-12-12 05:17:07 (3.23 MB/s) - ‘frontalimages_manuallyaligned_part1.zip’ saved [6209322/6209322]
    
    --2019-12-12 05:17:11--  https://fei.edu.br/~cet/frontalimages_manuallyaligned_part2.zip
    Resolving fei.edu.br (fei.edu.br)... 200.232.90.210
    Connecting to fei.edu.br (fei.edu.br)|200.232.90.210|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 6168379 (5.9M) [application/x-zip-compressed]
    Saving to: ‘frontalimages_manuallyaligned_part2.zip’
    
    frontalimages_manua 100%[===================>]   5.88M  3.02MB/s    in 1.9s    
    
    2019-12-12 05:17:13 (3.02 MB/s) - ‘frontalimages_manuallyaligned_part2.zip’ saved [6168379/6168379]
    


Unzip your zip files using:
```
!unzip <name_of_your_zip_file>
```


```python
!unzip frontalimages_manuallyaligned_part1.zip
!unzip frontalimages_manuallyaligned_part2.zip
```

    Archive:  frontalimages_manuallyaligned_part1.zip
      inflating: 178b.jpg                
      inflating: 179a.jpg                
      inflating: 179b.jpg                
      inflating: 180a.jpg                
      inflating: 180b.jpg                
      inflating: 181a.jpg                
      inflating: 181b.jpg                




```python
# importing all the necessary packages 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
```

###Vectorization
We will use Python Imaging Library (PIL) to load images from files.
`PIL.Image.open()` Opens and identifies the given image file.`Note` Open into greyscale, or L(Luminance) mode: 

```python
img = Image.open(f'{i}a.jpg').convert('L')  # i ranges from 1 to 200
img = img.resize((width, height),Image.ANTIALIAS) # for speed
```
Rearrange each Face image N x N into a column vector N<sup>2</sup> x 1. After Vectorization the shape of face matrix should be 200 x (width x height)






```python
# neutral face
neutral = []

for i in range(200):
    i += 1
    img = Image.open(f'{i}b.jpg').convert('L')
    img = img.resize((58,49), Image.ANTIALIAS)
    img2 = np.array(img).flatten() # vectorization
    neutral.append(img2)
```


```python
# Check face_matrix.shape 

faces_matrix = np.vstack(neutral)
faces_matrix.shape
```




    (200, 2842)




```python
# Find mean_face

mean_face = np.mean(faces_matrix, axis=0)
mean_face.shape
```




    (2842,)





```python
plt.imshow(mean_face.reshape(height,width),cmap='gray'); 
plt.title('Mean Face')
```





```python
plt.imshow(mean_face.reshape(49,58),cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f6fe62d8e80>




![png](output_12_1.png)


### Normalization


```python
#Perfrom Normalization
faces_norm = faces_matrix - mean_face
faces_norm.shape
```




    (200, 2842)




```python
# Calculate covariance matrix
#np.cov expects features as rows and observations as columns, so transpose
face_cov = np.cov(faces_norm.T)
face_cov.shape

```




    (2842, 2842)




```python
#Find Eigen Vectors and Eigen Values, you can use SVD from np.linalg.svd
eigen_vecs, eigen_vals, _ = np.linalg.svd(face_cov)
eigen_vecs.shape
```




    (2842, 2842)




Plot the first 10 Eigenfaces
```python
fig, axs = plt.subplots(1,3,figsize=(15,5))
for i in np.arange(10):
    ax = plt.subplot(2,5,i+1)
    img = <eigen_vectors>[:,i].reshape(height,width)
    plt.imshow(img, cmap='gray')
fig.suptitle("First 10 Eigenfaces", fontsize=16)
```




```python
#Plot the first 10 Eigenfaces
fig, axs = plt.subplots(1,3,figsize=(15,5))
for i in np.arange(10):
    ax = plt.subplot(2,5,i+1)
    img = eigen_vecs[:,i].reshape(49,58)
    plt.imshow(img, cmap='gray')
fig.suptitle("First 10 Eigenfaces", fontsize=16)
```




    Text(0.5, 0.98, 'First 10 Eigenfaces')




![png](output_18_1.png)



Reconstruction with increasing Eigenfaces
```
fig, axs = plt.subplots(2,5,figsize=(15,6))
for k, i in zip([0,1,9,19,39,79,159,199,399,799],np.arange(10)):
    # Reconstruct the first picture '1a.jpg' whose index is 0.
    # Get PC scores of the images (wights)
    # Reconstruct first face in dataset using k PCs (projected_face)
    ax = plt.subplot(2,5,i+1)
    ax.set_title("k = "+str(k+1))
    plt.imshow(projected_face.reshape(height,width)+mean_face.reshape(height,width),cmap='gray');
```




```python
# Reconstruct with increasing Eigenfaces
fig, axs = plt.subplots(2,5,figsize=(15,6))
for k, i in zip([0,1,9,19,39,79,159,199,399,799],np.arange(10)):
    # Reconstruct the first picture '1a.jpg' whose index is 0.
    weight = faces_norm[0,:].dot(eigen_vecs[:,:k]) # Get PC scores of the images
    projected_face = weight.dot(eigen_vecs[:,:k].T) # Reconstruct first face in dataset using k PCs
    ax = plt.subplot(2,5,i+1)
    ax.set_title("k = "+str(k+1))
    plt.imshow(projected_face.reshape(49,58)+mean_face.reshape(49,58),cmap='gray');
fig.suptitle(("Reconstruction with Increasing Eigenfaces"), fontsize=16)
```




    Text(0.5, 0.98, 'Reconstruction with Increasing Eigenfaces')




![png](output_20_1.png)



```python

```
