#Isaac Jorgensen
#4/16/2019
#Linear Regression techniques used to identfy diabetics in the Pima Indians Diabetes dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt

#the complete set of data, nothing will be removed
data = pd.read_excel('pima-indians-diabetes.xlsx','pima-indians-diabetes')
tally = {20:0, 40:0, 60:0, 80:0, 100:0}

for exp in range(1,1000):
    # N=768 data samples
    for n in [20,40,60,80,100]:
        
       #split the data set into diabetes and no-diabetes
        diabetes = data[data['Outcome'] == 1]
        nodiabetes = data[data['Outcome'] == 0]

        #sample n from each set
        diabetes_samples = diabetes.sample(n, axis=0)
        nodiabetes_samples = nodiabetes.sample(n, axis=0)

        #remove the samples from the original sets
        diabetes = diabetes.drop(diabetes_samples.index, axis=0)
        nodiabetes = nodiabetes.drop(nodiabetes_samples.index, axis=0)

        #concatenate the diabetes and nodiabetes for the training set and the test set
        X_train = pd.concat([diabetes_samples, nodiabetes_samples], ignore_index=True)
        X_test = pd.concat([diabetes, nodiabetes], ignore_index=True)

        #remove the 'Outcome' from both the training and test sets
        t_train = pd.concat([diabetes_samples['Outcome'], nodiabetes_samples['Outcome']], ignore_index=True).to_numpy()
        X_train = X_train.drop(['Outcome'], axis=1).to_numpy()

        t_test = pd.concat([diabetes['Outcome'], nodiabetes['Outcome']], ignore_index=True).to_numpy()
        X_test = X_test.drop(['Outcome'], axis=1).to_numpy()
        
        #convert t_train, X_train, t_test, and X_test to tensors
        t_train = tf.constant(t_train, dtype=tf.float64, shape=[2*n, 1], name = "t_train")
        X_train = tf.constant(X_train, dtype=tf.float64, shape=[2*n, 8], name = "X_train")
        t_test = tf.constant(t_test, dtype=tf.float64, shape=[768-2*n, 1], name = "t_test")
        X_test = tf.constant(X_test, dtype=tf.float64, shape=[768-2*n, 8], name = "X_test")

        #transpose X
        XT_train = tf.transpose(X_train)

        #perform (XT*X)^-1 * XT * t = w
        w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT_train,X_train)), XT_train), t_train)

        #multiply w (8 x 1)by a matrix X' (768-2n x 8)
        t_hat = tf.matmul(X_test, w)
        #result is t_hat, compare it to t_test and add correct amount to a tally for that n

        with tf.Session() as sess:
            t_train_val = t_train.eval()
            X_train_val = X_train.eval()
            XT_train_val = X_train.eval()
            
            w_val = w.eval()
            
            t_test_val = t_test.eval()
            X_test_val = X_test.eval()            
            
            t_hat_val = t_hat.eval()
            
            #change every t_hat/arr value >= 0.5 -> 1 and every value <0.5 -> 0
            for i in range(len(t_hat_val)):
                if t_hat_val[i] >= 0.5: 
                    t_hat_val[i] = 1
                else:
                    t_hat_val[i] = 0
                #if the predicted outcome and the actual outcome match, increase the tally
                if t_hat_val[i] == t_test_val[i]: tally[n] += 1

        #print("Working on training size" + str(n))
    if exp%50 == 0: 
        print("Experiment #" + str(exp))
        print(tally)

perc = {20:0, 40:0, 60:0, 80:0, 100:0}
for m in tally:
    perc[m] = tally[m]/((768-2*m)*exp)
    print(str(m) + ": " + str(perc[m]))

plt.plot(list(perc.keys()), list(perc.values()))
plt.show()
