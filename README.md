# Py-AutoML


[![LICENCE.md](https://img.shields.io/github/license/PrudhviGNV/py-automl)](https://github.com/PrudhviGNV/py-automl/blob/master/LICENCE.md)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/PrudhviGNV/py-automl)
[![Website prudhvignv.github.io](https://img.shields.io/website-up-down-green-red/https/naereen.github.io.svg)](https://prudhvignv.github.io/)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/PrudhviGNV)<br/>
[![PyPI version fury.io](https://badge.fury.io/py/py-automl.svg)](https://pypi.python.org/pypi/py-automl/)
[![PyPI format](https://img.shields.io/pypi/format/ansicolortags.svg)](https://pypi.python.org/pypi/py-automl/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/py-automl.svg)](https://pypi.python.org/pypi/py-automl/)
[![PyPI status](https://img.shields.io/pypi/status/py-automl.svg)](https://pypi.python.org/pypi/py-automl/) <br/>
[![Open Source Love svg2](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/PrudhviGNV/open-source-badges/)
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/PrudhviGNV/badges)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/PrudhviGNV/)
[![ForTheBadge built-by-prudhvignv](http://ForTheBadge.com/images/badges/built-by-developers.svg)](https://GitHub.com/PrudhviGNV/)






  



# Introduction

## What is Py-AutoML?
Py-AutoML is an open source `low-code` machine learning library in Python that aims to reduce the hypothesis to insights cycle time in a ML experiment. It mainly helps to do our pet projects quickly and efficiently. In comparison with the other open source machine learning libraries, Py-AutoML is an alternative low-code library that can be used to perform complex machine learning tasks with only few lines of code. Py-AutoML is essentially a Python wrapper around several machine learning libraries and frameworks such as `scikit-learn`, 'tensorflow','keras' and many more. 

The design and simplicity of Py-AutoML is inspired by the  two principles KISS (keep it simple and sweet) and DRY (Don't Repeat Yourself) . We as engineers have to find a way  effective way to mitigate this gap and address data related challenges in business setting.


# Modules
Py-AutoML is a minimalistic library which not  simplifies the machine learning tasks and also makes our work easier.

Py-AutoML consists of so many functionalities. such as 
-----------------

   - #### model.py- implementing popular neural networks such as googlenet , vgg16, simple cnn ,basic cnn, lenet5, alexnet, lstm, mlp etc..
   - #### checkpoint.py - consists of callbacks function which is used to store metrics 
   - #### utils.py - consists of some functionalities used to preprocess test images, spliting the data.
   - #### preprocess.py - used to preprocess image dataset such as resize, reshape, convert to greyscale, normalisation etc..
   - #### ml.py - allow us to implement and check metrics of popular classical machine learning models such as random forest, decision tree, svm , logistic regression and also displays metric reports of every model
   - #### visualize.py - allow us to visualize neural networks in pictorial and graphs form.
   
   
 # ml.py -> Implemented algorithms

------------
- ### Logistic Regression
- ### Support Vector Machine
- ### Decision Tree Classifier
- ### Random Forest Classifier
- ### K-Nearest Neighbors
--------------------------

   
 # model.py -> Implemented popular neural network architectures

------------
- ### GoogleNet
- ### VGG16
- ### AlexNet
- ### Lenet5
- ### Inception
- ### simple & basic cnn
- ### basic_mlp & deep_mlp
- ### lstm
with predefined configurations
--------------------------
# Getting started

-----------------

## Install the package
```bash
pip install py-automl
```
Navigate to folder and install requirements: 
```bash
pip install -r requirements.txt

```

## Usage
Importing the package
```python
import pyAutoML
from pyAutoML import *
from pyAutoML.model import *
# like that...
```
Assign the variables X and Y to the desired columns and assign the variable size to the desired test_size.  
```python
X = < df.features >
Y = < df.target >
size = < test_size >
```
## Encoding Categorical Data 
Encode target variable if non-numerical:
```python
from pyAutoML import *
Y = EncodeCategorical(Y)
```
## Running py-automl

signature is as follows :   ML(X, Y, size=0.25, *args)
```python
from pyAutoML.ml import ML,ml, EncodeCategorical

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets




##reading the Iris dataset into the code
df =  datasets.load_iris()

##assigning the desired columns to X and Y  in preparation for running fastML
X = df.data[:, :4]
Y = df.target

##running the EncodeCategorical function from fastML to handle the process of categorial encoding of data
Y = EncodeCategorical(Y)
size = 0.33

ML(X, Y, size, SVC(), RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier(), LogisticRegression(max_iter = 7000))

```
### output
```python
____________________________________________________
.....................Py-AutoML......................
____________________________________________________
SVC ______________________________ 

Accuracy Score for SVC is 
0.98


Confusion Matrix for SVC is 
[[16  0  0]
 [ 0 18  1]
 [ 0  0 15]]


Classification Report for SVC is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      0.95      0.97        19
           2       0.94      1.00      0.97        15

    accuracy                           0.98        50
   macro avg       0.98      0.98      0.98        50
weighted avg       0.98      0.98      0.98        50



____________________________________________________
RandomForestClassifier ______________________________ 

Accuracy Score for RandomForestClassifier is 
0.96


Confusion Matrix for RandomForestClassifier is 
[[16  0  0]
 [ 0 18  1]
 [ 0  1 14]]


Classification Report for RandomForestClassifier is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       0.95      0.95      0.95        19
           2       0.93      0.93      0.93        15

    accuracy                           0.96        50
   macro avg       0.96      0.96      0.96        50
weighted avg       0.96      0.96      0.96        50



____________________________________________________
DecisionTreeClassifier ______________________________ 

Accuracy Score for DecisionTreeClassifier is 
0.98


Confusion Matrix for DecisionTreeClassifier is 
[[16  0  0]
 [ 0 18  1]
 [ 0  0 15]]


Classification Report for DecisionTreeClassifier is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      0.95      0.97        19
           2       0.94      1.00      0.97        15

    accuracy                           0.98        50
   macro avg       0.98      0.98      0.98        50
weighted avg       0.98      0.98      0.98        50



____________________________________________________
KNeighborsClassifier ______________________________ 

Accuracy Score for KNeighborsClassifier is 
0.98


Confusion Matrix for KNeighborsClassifier is 
[[16  0  0]
 [ 0 18  1]
 [ 0  0 15]]


Classification Report for KNeighborsClassifier is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      0.95      0.97        19
           2       0.94      1.00      0.97        15

    accuracy                           0.98        50
   macro avg       0.98      0.98      0.98        50
weighted avg       0.98      0.98      0.98        50



____________________________________________________
LogisticRegression ______________________________ 

Accuracy Score for LogisticRegression is 
0.98


Confusion Matrix for LogisticRegression is 
[[16  0  0]
 [ 0 18  1]
 [ 0  0 15]]


Classification Report for LogisticRegression is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      0.95      0.97        19
           2       0.94      1.00      0.97        15

    accuracy                           0.98        50
   macro avg       0.98      0.98      0.98        50
weighted avg       0.98      0.98      0.98        50



                    Model Accuracy
0                     SVC     0.98
1  RandomForestClassifier     0.96
2  DecisionTreeClassifier     0.98
3    KNeighborsClassifier     0.98
4      LogisticRegression     0.98
```

### you can also write as follows
```python
ML(X,Y)
```
### output
```python
____________________________________________________
.....................Py-AutoML......................
____________________________________________________
SVC ______________________________ 

Accuracy Score for SVC is 
0.9736842105263158


Confusion Matrix for SVC is 
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]


Classification Report for SVC is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        13
           1       1.00      0.94      0.97        16
           2       0.90      1.00      0.95         9

    accuracy                           0.97        38
   macro avg       0.97      0.98      0.97        38
weighted avg       0.98      0.97      0.97        38



____________________________________________________
RandomForestClassifier ______________________________ 

Accuracy Score for RandomForestClassifier is 
0.9736842105263158


Confusion Matrix for RandomForestClassifier is 
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]


Classification Report for RandomForestClassifier is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        13
           1       1.00      0.94      0.97        16
           2       0.90      1.00      0.95         9

    accuracy                           0.97        38
   macro avg       0.97      0.98      0.97        38
weighted avg       0.98      0.97      0.97        38



____________________________________________________
DecisionTreeClassifier ______________________________ 

Accuracy Score for DecisionTreeClassifier is 
0.9736842105263158


Confusion Matrix for DecisionTreeClassifier is 
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]


Classification Report for DecisionTreeClassifier is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        13
           1       1.00      0.94      0.97        16
           2       0.90      1.00      0.95         9

    accuracy                           0.97        38
   macro avg       0.97      0.98      0.97        38
weighted avg       0.98      0.97      0.97        38


____________________________________________________
KNeighborsClassifier ______________________________ 

Accuracy Score for KNeighborsClassifier is 
0.9736842105263158


Confusion Matrix for KNeighborsClassifier is 
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]


Classification Report for KNeighborsClassifier is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        13
           1       1.00      0.94      0.97        16
           2       0.90      1.00      0.95         9

    accuracy                           0.97        38
   macro avg       0.97      0.98      0.97        38
weighted avg       0.98      0.97      0.97        38



____________________________________________________
LogisticRegression ______________________________ 

Accuracy Score for LogisticRegression is 
0.9736842105263158


Confusion Matrix for LogisticRegression is 
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]


Classification Report for LogisticRegression is 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        13
           1       1.00      0.94      0.97        16
           2       0.90      1.00      0.95         9

    accuracy                           0.97        38
   macro avg       0.97      0.98      0.97        38
weighted avg       0.98      0.97      0.97        38



                    Model            Accuracy
0                     SVC  0.9736842105263158
1  RandomForestClassifier  0.9736842105263158
2  DecisionTreeClassifier  0.9736842105263158
3    KNeighborsClassifier  0.9736842105263158
4      LogisticRegression  0.9736842105263158
```

   
 ## Defining popular neural networks
 
 ### implementing alexNet may looks like this
 
 ```python
  #Instantiation
    AlexNet = Sequential()

    #1st Convolutional Layer
    AlexNet.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #2nd Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #3rd Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    #4th Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    #5th Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #Passing it to a Fully Connected layer
    AlexNet.add(Flatten())
    # 1st Fully Connected Layer
    AlexNet.add(Dense(4096, input_shape=(32,32,3,)))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    AlexNet.add(Dropout(0.4))

    #2nd Fully Connected Layer
    AlexNet.add(Dense(4096))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #3rd Fully Connected Layer
    AlexNet.add(Dense(1000))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #Output Layer
    AlexNet.add(Dense(10))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation(classifier_function))

    AlexNet.compile('adam', loss_function, metrics=['acc'])
    return AlexNet
```
But we implement this in a single line of code like below using this package.
```python
alexNet_model = model(input_shape= (30,30,4) , arch="alexNet", classify="Mulit" )
```
Similarly we can also implement
```python
alexNet_model = model("alexNet")

lenet5_model = model("lenet5")

googleNet_model = model("googleNet")

vgg16_model = model("vgg16")

### etc...

```
For more generalization , let's observe following code.
```python
# Lets take all models that are defined in the py_automl and which are implemented in a signle line of code
models = ["simple_cnn", "basic_cnn", "googleNet", "inception","vgg16","lenet5","alexNet", "basic_mlp","deep_mlp","basic_lstm","deep_lstm" ]

d= {}

for i in models:
  d[i] = model(i)  # assigning all architectures to its model names using dictionary
  
```

## Visualization 
### we can visualize neural networks architecture in different forms with ease.
Let's observe the following code for better understanding
```python
import keras
from keras import layers
model = keras.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=10, activation = 'softmax'))
```
now let's visualise this
```python 
nn_visualize(model)
```
By default , it returns keras visualization object
### output:
![i1](https://user-images.githubusercontent.com/39909903/91040097-840bbf80-e5c2-11ea-8c3d-fad294b20722.png)


```python

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



#Neural network visualization 

nn_visualize(model,type = "graphviz")

```
### output
![1_gTwmrLh1aYLzayMylHGIeg](https://user-images.githubusercontent.com/39909903/91041224-8242fb80-e5c4-11ea-8539-4c2c35f7bab5.jpeg)


This library is so developer friendly that even we declare type with starting letters.
```python
from pyAutoML.model import *
model2 = model(arch="alexNet")

nn_visualize(model2,type="k")

```
### output:
![i3](https://user-images.githubusercontent.com/39909903/91040108-8837dd00-e5c2-11ea-87c4-a9951804d3c8.png)

## This is a minimal documentation about the package. <br/>
For more information and understanding, see examples [HERE](https://github.com/PrudhviGNV/py-automl/edit/master/examples)
and source code: [GITHUB](https://github.com/PrudhviGNV/py-automl)
-------

## Author: [Prudhvi GNV](prudhvignv.github.io)
-------
# Contact:
<br/>
[LinkedIn](https://linkedin.com/in/prudhvignv/) <br/>
[Github](https://github.com/PrudhviGNV) <br/>
[Instagram](https://instagram.com/prudhvi-gnv)




 

