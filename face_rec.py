#Isaac Jorgensen
#4/26/2019
#Principle Component Analysis used for facial recognition.

#facial recognition
import numpy as np
import pandas as pd
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
import time

#X_train will hold the 10304 pixel values for each of 60 training images
X_train = np.zeros((10304, 0))
#train_label will keep track of the subject that the training face belongs to
train_label = []
#X_test will hold the 10304 pixel values for each of 40 testing images
X_test = np.zeros((10304, 0))
#test_label will keep track of the subject that the test face belongs to
test_label = []

#X_bar will hold the zero centered version of each training image
X_bar = np.zeros((10304, 60))

K= {1:0 ,2:0 ,3:0 ,6:0 ,10:0 ,20:0 , 30:0 , 50:0 }

#load each image and place it in either the training or test matrices, keep track of
#the subject they belong two as well
for i in range(1,10+1):
    for j in range(1,10+1):
        
        #prepare image data
        x = np.zeros((10304, 1))
        filepath = "/Users/isaacjorgensen/Documents/SCU/SCU Classes/COEN 240/Homework2/att_faces_10/s%s/%s.pgm" % (str(i),str(j))
        im = Image.open(filepath)
        np_im = np.array(im)

        #reshape the newly loaded image from 112x92 into 10304x1, this appends each column to the bottom of
        #its predecessor, creating the 1D array of each image
        x = np.reshape(np_im, (10304,1), order = 'F')
        
        #based on the photo number for each subject, add it to the train or test matrix
        if j == 1 or j == 3 or j == 4 or j == 5 or j == 7 or j == 9:
            X_train = np.append(X_train,x, axis = 1)
            train_label = np.append(train_label, i)
        else:
            X_test = np.append(X_test,x, axis = 1)
            test_label = np.append(test_label, i)


#compute the average, mu, of all of the images in the training set
mu = X_train.mean(1)

#subtract the average from each image in the trainin set to create X_bar
for i in range(np.size(X_train,1)):
    X_bar[:, i] = X_train[:, i] - mu

#calculate the Priciple Components, U
U, S, VT = np.linalg.svd(X_bar, full_matrices=True)

#calculates the y column vector in X_train that y is closest to and return its index
def minDist(y):
    dist = np.zeros(60)
    for i in range(np.size(X_train,1)):
        dist[i] = np.linalg.norm(y - y_train[:,i])
    return dist.argmin()

#prints each image and its predicted counterpart
def printimg():
    filepath = "/Users/isaacjorgensen/Documents/SCU/SCU Classes/COEN 240/Homework2/att_faces_10/s%s/1.pgm" % (str(int(train_label[classification])))
    im = Image.open(filepath)
    im.show()
    filepath = "/Users/isaacjorgensen/Documents/SCU/SCU Classes/COEN 240/Homework2/att_faces_10/s%s/1.pgm" % (str(int(test_label[i])))
    im = Image.open(filepath)
    im.show()
    time.sleep(2)

#calculate y_train, the y values of every training image
#calculate y for each test image and call minDist to determine
#how to classify the image from which y was created
for k in K:
    y_train = np.matmul((U[:, 0:k].transpose()), X_bar)
    for i in range(np.size(X_test,1)):
        #y = UT*(x-mu) where UT is dimension kx10304
        y = np.matmul((U[:, 0:k].transpose()),(X_test[:, i] - mu))
        #do argmin here
        classification = minDist(y)
        #keep track of correct classifications
        if train_label[classification] == test_label[i]: K[k] += 1
        printimg()
    K[k] = K[k]/40

print(K)
#plt.plot(list(K.keys()), list(K.values()))
#plt.show()
