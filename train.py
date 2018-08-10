#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 23:04:39 2017

@author: fjiang
"""
import os
import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from random import shuffle


#read in csv file  
lines = []
path = 'data/driving_log.csv'
with open(path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#split trainging data and validation data
#from sklearn.model_selection import train_test_split
#train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#-----------------------------------------------------------------------------
#never use generator. THe regular workflow works well and fast in my computer.
#def generator(samples, batch_size=32):
#    num_samples = len(samples)
#    while 1:
#        shuffle(samples)
#        for offset in range(0,num_samples,batch_size):
#            batch_samples = samples[offset:offset+batch_size]
#            
#            images = []
#            angles = []
#            
#            for batch_sample in batch_samples:
#                name = '/data/IMG/' + batch_sample[0].split('/')[-1]
#                center_image = cv2.imread(name)
#                center_angle = float(batch_sample[3])
#                images.append(center_image)
#                angles.append(center_angle)
#                
#            X_train = np.array(images)
#            y_train = np.array(angles)
#            yield sklearn.utils.shuffle(X_train, y_train)
#            
#train_generator = generator(train_samples, batch_size = 32)           
#validation_generator = generator(validation_samples, batch_size = 32)           

#-----------------------------------------------------------------------------
# May use left and right camera 
#    for row in reader:
#       steering_center = float(row[3])
        
#        correction = 0.2
#        steering_left = steering_center + correction
#        steering_right = steering_center - correction

#        directory = "data/IMG/"
#        img_center = process_image(np.asarray(Image.open(directory + row[0])))
#        img_left   = process_image(np.asarray(Image.open(directory + row[1])))
#        img_right  = process_image(np.asarray(Image.open(directory + row[2])))

#        car_images.extend(img_center, img_left, img_right)
#        steering_angles.extend(steering_center, steering_left, steering_right)

#original train method
#read in image file 
images=[]
measurements = []
for line in lines:
    #read in middle, left, right images as training data and 
    #steering angle as labels.
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        #convert BGR to RGB
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)


#augmentation images and labels for better trainging data information
augmented_images, augmented_measurements = [],[]
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*(-1.0))

#Output original image and augmented image
#Try use PIL
nimage = images.count
from PIL import Image
for i, image in enumerate(augmented_images):
    if i==0:
        image1 = Image.fromarray(image)
        image1.save("image0.png")
        image1 = cv2.flip(image,1)
        image1 = Image.fromarray(image1)
        image1.save("image0_flip.png")
    if i==1000:
        image1 = Image.fromarray(image)
        image1.save("image1.png")
        image1 = cv2.flip(image,1)
        image1 = Image.fromarray(image1)
        image1.save("image1_flip.png")
    if i==2000:
        image1 = Image.fromarray(image)
        image1.save("image2.png")
        image1 = cv2.flip(image,1)
        image1 = Image.fromarray(image1)
        image1.save("image2_flip.png")
    if i==3000:
        image1 = Image.fromarray(image)
        image1.save("image3.png")
        image1 = cv2.flip(image,1)
        image1 = Image.fromarray(image1)
        image1.save("image3_flip.png")



#assign trainging data
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Flatten, Dense, Lambda, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
#from keras.layers.pooling import MaxPooling2D

#use keras for CNN
model = Sequential()

#Lambda layer as additional layer plus normalization
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160,320,3), 
                 output_shape=(160,320,3)))

#Crop image to throwout useless information
model.add(Cropping2D(cropping=((70,25),(0,0))))

#Multiple convolution layer, referenced from LeNet
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#Before trainging, configure the learnign process
model.compile(loss='mse', optimizer='adam')

#Print trainging result
history_object = model.fit(X_train, y_train, nb_epoch=3, validation_split=0.3, 
         shuffle=True)

#model.fit_generator(train_generator, samples_per_epoch=len(train_samples), 
#                    validation_data = validation_generator, nb_val_samples=len(validation_samples), 
#                    nb_epoch=3)

print(history_object.history.keys)

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model MSE loss')
plt.xlabel('MSE loss')
plt.xlabel('epoch')
plt.legend(['trainging set', 'validation set'], loc='upper right')
plt.show()

#Save trained model
model.save('model.h5')
exit()

    