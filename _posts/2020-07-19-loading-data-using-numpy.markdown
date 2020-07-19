---
layout: post
title:  "Loading and saving images directory in numpy"
date:   2020-07-19 14:55:04 +0545
categories: jekyll blog

---
Connect your colab to local runtime


### Purpose of the blog:
Load data similar to MNIST style dataset. This is particulary useful to feed your data directoly in many ML pipeline such as scikit-learn. 

```
import cv2
import numpy as np
class_label = {'male':0, 'female': 1}
train_dir = os.path.join(DATASET_PATH, 'Training')
valid_dir = os.path.join(DATASET_PATH, 'Validation')


def datagen(output_size):
  X, y = [], []  
   
  for cat in os.listdir(train_dir):
    cat_dir = train_dir + '/' + cat
    for img_file in os.listdir(cat_dir):
      img_path = cat_dir + '/' + img_file
      img = cv2.imread(img_path)
      img = cv2.resize(img, dsize=(output_size, output_size), interpolation=cv2.INTER_CUBIC)
     
      X.append(img)
      y.append(class_label[str(cat)])

  for cat in os.listdir(valid_dir):
    cat_dir = valid_dir + '/' + cat
    for img_file in os.listdir(cat_dir):
      img_path = cat_dir + '/' + img_file
      img = cv2.imread(img_path)
      img = cv2.resize(img, dsize=(output_size, output_size), interpolation=cv2.INTER_CUBIC)
      X.append(img)
      y.append(class_label[str(cat)])   
  X = np.array(X)
  y = np.array(y)
  print(X.shape)
  print(y.shape)

  return X, y    
  
X , y =  datagen(output_size = 64)

X = X.astype("float") / 255.0


np.save(saveFile_X, X)
np.save(saveFile_y, y)

```
## Load the saved data

```
X = np.load(saveFile_X)
y = np.load(saveFile_y)
```

## Split your dataset using scikit learn

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
```
Note that stratify is useful as it maintains the class proportions during split.



