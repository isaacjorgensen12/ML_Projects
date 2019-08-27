#Isaac Jorgensen
#5/27/19
#Logistic Regression employed to identify handwritten numbers.

#Logistic Regression - Number Recognition

import tensorflow as tf
mnist = tf.keras.datasets.mnist

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from functionToCall import plot_confusion_matrix

#load and prepare the data for processing
(x_traino, y_train),(x_testo, y_test) = mnist.load_data()
x_train = np.reshape(x_traino,(60000,28*28))
x_test = np.reshape(x_testo,(10000,28*28))
x_train, x_test = x_train / 255.0, x_test / 255.0

#use sklearn's LogisticRegression function to create the model
logreg = LogisticRegression(solver='saga', multi_class='multinomial',max_iter = 100,verbose=2)

#fit the training data
logreg.fit(x_train, y_train)

#form a prediction from the model made with the training data
y_test_hat = logreg.predict(x_test)

#tally the number of correct predictions
num_correct = 0
for i in range(len(y_test)):
    if y_test_hat[i]==y_test[i]:
        num_correct +=1
        
Accuracy_rate = num_correct/len(y_test)
print("Accuracy Rate = ", Accuracy_rate)


#setup and plot the confusion matrix
confusion_matrix(y_test, y_test_hat, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
class_names = np.asarray(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype=np.int)
plot_confusion_matrix(y_test, y_test_hat, classes=class_names,
                      title='Confusion matrix, without normalization')

plot_confusion_matrix(y_test, y_test_hat, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()