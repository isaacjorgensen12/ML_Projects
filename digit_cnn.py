#Isaac Jorgensen
#5/31/19
#A 9-layer convolutional neural net to identify handwritten digits.

#9-Layer CNN - Digit Recognition

import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from functionToCall import plot_confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn import metrics

#load and split the data into training and test
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)) 
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#create the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) #2.  the 1st 2d-convolutional layer
model.add(layers.MaxPooling2D(pool_size=(2, 2))) #3. 1st max pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1))) #4.  the 2nd 2d-convolutional layer
model.add(layers.MaxPooling2D(pool_size=(2, 2))) #5. 2nd max pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1))) #6.  the 3rd 2d-convolutional layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu')) #7. the 1st fully connected layer
model.add(layers.Dense(64, activation='relu')) #8. the 2nd fully connected layer
model.add(layers.Dense(10, activation='softmax')) #9. the fully connected output layer

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(train_images, train_labels, epochs=5, batch_size=35,  verbose=2)

# calculate predictions
predictions = model.predict(test_images)

#to hold all of the rounded values
test_images_hat = np.zeros((np.size(predictions, 0), np.size(predictions, 1)))
#to hold the final predictions
final_pred = np.zeros((np.size(predictions, 0)))

#round all of the predictions to 0 or 1
for i in range(np.size(predictions, 0)):
    for j in range(np.size(predictions, 1)):
        test_images_hat[i,j] = round(predictions[i,j])

#tally the number of correct predictions
num_correct = 0
for i in range(len(test_images)):
    if 1 in test_images_hat[i,:]:
        if test_images_hat[i,:].tolist().index(1) == test_labels[i]:
            num_correct +=1

#translate the predictions to index numbers for the final_pred i.e. 000010000 -> 4, 000000100 -> 7
#if a data sample has no predicted number, it's set to 0 (this is caused if no values are higher than 0.5)
for i in range(len(test_images_hat)):
    if 1 in test_images_hat[i,:]:
        final_pred[i] = test_images_hat[i,:].tolist().index(1)
    else:
        final_pred[i] = 0

#calculate the accuracy rate
Accuracy_rate = num_correct/len(test_images)
print("Accuracy Rate = ", Accuracy_rate)

#plot the confusion matrix
confusion_matrix(test_labels, final_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
class_names = np.asarray(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype=np.int)
plot_confusion_matrix(test_labels, final_pred, classes=class_names,title='Confusion matrix, without normalization')

plot_confusion_matrix(test_labels, final_pred, classes=class_names, normalize=True,title='Normalized confusion matrix')

plt.show()

#Accuracy Rate =  0.9915