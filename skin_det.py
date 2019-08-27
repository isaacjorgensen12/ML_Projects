#Isaac Jorgensen
#5/20/19
#Bayesian Decision Theory used for skin detection in images.

#Skin Recognition
from __future__ import division
import numpy as np
import pandas as pd
import sys
import cv2
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

#load each image
img_filepath = "/Users/isaacjorgensen/Documents/SCU/SCU Classes/COEN 240/family.jpg"
mask_filepath = "/Users/isaacjorgensen/Documents/SCU/SCU Classes/COEN 240/family.png"

im = cv2.imread(img_filepath)
mask_im = cv2.imread(mask_filepath)

np_im = np.array(im).astype(float)
np_mask = np.array(mask_im).astype(float)
#change each image from 250x200x3/4 into 3/4x50000

x = np_im.transpose(2,0,1).reshape(3,-1)
x_mask = np_mask.transpose(2,0,1).reshape(3,-1)
x_copy = x_mask #to be altered later

#load each image
img_filepath = "/Users/isaacjorgensen/Documents/SCU/SCU Classes/COEN 240/portrait.jpg"
mask_filepath = "/Users/isaacjorgensen/Documents/SCU/SCU Classes/COEN 240/portrait.png"

im = cv2.imread(img_filepath)
mask_im = cv2.imread(mask_filepath)

np_im = np.array(im).astype(float)
np_mask = np.array(mask_im).astype(float)

#change each image from 250x200x3/4 into 3/4x50000
x_test = np_im.transpose(2,0,1).reshape(3,-1)
x_mask_test = np_mask.transpose(2,0,1).reshape(3,-1)
x_copy_test = x_mask_test #to be altered later

#Scale R and G values for normal image
for i in range(np.size(x, 1)):
    r = x[0,i]
    g = x[1,i]
    b = x[2,i]
    if r+g+b == 0:
        x[0,i] = float(1/3)
        x[1,i] = float(1/3)
    else:
        x[0,i] = r/(r+g+b)
        x[1,i] = g/(r+g+b)
x = np.delete(x,(2), axis=0)

#Scale R and G values for normal test image
for i in range(np.size(x_test, 1)):
    r = x_test[0,i]
    g = x_test[1,i]
    b = x_test[2,i]
    if r+g+b == 0:
        x_test[0,i] = float(1/3)
        x_test[1,i] = float(1/3)
    else:
        x_test[0,i] = r/(r+g+b)
        x_test[1,i] = g/(r+g+b)

x_test = np.delete(x_test,(2), axis=0)


skin_count = 0
background_count = 0
mean_rone = 0.0
mean_gone = 0.0
mean_rzero = 0.0
mean_gzero = 0.0

#calculate the mean r and mean g for skin or background, calculate the total number of skin/background pixels
for i in range(np.size(x_mask, 1)):
    if x_mask[0,i] != 0: #skin
        mean_rone += x[0,i]
        mean_gone += x[1,i]
        skin_count += 1
    else: #background
        mean_rzero += x[0,i]
        mean_gzero += x[1,i]
        background_count += 1
#one = skin, zero = background

mean_rone /= skin_count
mean_gone /= skin_count
mean_rzero /= background_count
mean_gzero /= background_count

var_rone = 0.0
var_gone = 0.0
var_rzero = 0.0
var_gzero = 0.0


#calculate the variance of r and variance of g for skin or background
for i in range(np.size(x_mask, 1)):
    if x_mask[0,i] != 0: #skin
        var_rone += (x[0,i] - mean_rone)**2
        var_gone += (x[1,i] - mean_gone)**2
    else: #background
        var_rzero += (x[0,i] - mean_rzero)**2
        var_gzero += (x[1,i] - mean_gzero)**2
#one = skin, zero = background
var_rone /= skin_count
var_gone /= skin_count
var_rzero /= background_count
var_gzero /= background_count

pHzero = 1.0
pHzerok = np.zeros(np.size(x_test, 1)).astype(float)
pHone = 1.0
pHonek = np.zeros(np.size(x_test, 1)).astype(float)


#calculate the joint probabilities for H0(background) and H1(skin)
for i in range(np.size(x_test, 1)):
        pHonek[i] = ((1/(np.sqrt(2*np.pi)*np.sqrt(var_rone)))*(np.e**(-0.5*(((x_test[0,i]-mean_rone)**2)/var_rone))))*((1/(np.sqrt(2*np.pi)*np.sqrt(var_gone)))*(np.e**(-0.5*(((x_test[1,i]-mean_gone)**2)/var_gone))))
        #pHone *= pHonek[i]
        pHzerok[i] = ((1/(np.sqrt(2*np.pi)*np.sqrt(var_rzero)))*(np.e**(-0.5*(((x_test[0,i]-mean_rzero)**2)/var_rzero))))*((1/(np.sqrt(2*np.pi)*np.sqrt(var_gzero)))*(np.e**(-0.5*(((x_test[1,i]-mean_gzero)**2)/var_gzero))))
        #pHzero *= pHzerok[i]

pH0 = background_count / (background_count + skin_count)
pH1 = skin_count / (background_count + skin_count)


for i in range(np.size(x_copy_test, 1)):
    if pHonek[i]/pHzerok[i] < pH0/pH1:
        #background
        x_copy_test[0,i] = 0
        x_copy_test[1,i] = 0
        x_copy_test[2,i] = 0
    else:
        #skin
        x_copy_test[0,i] = 255
        x_copy_test[1,i] = 255
        x_copy_test[2,i] = 255


actual_background = 0
actual_skin = 0
predicted_background = 0
predicted_skin = 0
false_background = 0
false_skin = 0
for i in range(np.size(x_copy_test, 1)): 
    if x_mask_test[0,i]==0:
        actual_background += 1
        if x_copy_test[0,i] == x_mask_test[0,i]:
            predicted_background +=1
        else:
            false_background += 1
    else:
        actual_skin += 1
        if x_copy_test[0,i] == x_mask_test[0,i]:
            predicted_skin +=1
        else:
            false_skin += 1
print(predicted_skin/actual_skin)
print(predicted_background/actual_background)
print(false_background/actual_background)
print(false_skin/actual_skin)

final_x = x_copy_test.reshape(250,200,3).transpose(0,1,2)

final_img = Image.fromarray(final_x.astype('uint8'))
final_img.save("result.png")