import os
import csv
import sys



# Import data
samples = []
with open('data/driving_log.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    # Skip header
    next(csvreader)
    for line in csvreader:
        samples.append(line)


# Separate in training and validation
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Define generator
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import sklearn
from sklearn.utils import shuffle

def generator(samples, batch_size=1):
    num_samples = len(samples)
    

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # load center image
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB);

                # load left image
                name = 'data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB);

                # load right image
                name = 'data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB);

                # Load angle of center image
                center_angle = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                # Crop images
                h, w, d = center_image.shape
                cropTop = 50
                cropBot = 25
                center_crop = center_image[cropTop:h-cropBot, :, :]
                left_crop = left_image[cropTop:h-cropBot, :, :]
                right_crop = right_image[cropTop:h-cropBot, :, :]

                # Normalize images and mean center
                # center_norm = center_crop / 127.5 - 1
                # left_norm = left_crop / 127.5 - 1
                # right_norm = right_crop / 127.5 - 1

                # Append cropped RGB center, left and right images
                # images.append(center_norm)
                # angles.append(center_angle)
                # images.append(left_norm)
                # angles.append(left_angle)
                # images.append(right_norm)
                # angles.append(right_angle)                
                images.append(center_crop)
                images.append(left_crop)
                images.append(right_crop)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

                # Append flipped center image
                #image_flip = cv2.flip(center_image, 1)
                image_flip = np.fliplr(center_crop)
                images.append(image_flip)
                angles.append(-center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)




# Use generator
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=1)
validation_generator = generator(validation_samples, batch_size=1)

ch, row, col = 3, 80, 320  # Trimmed image format

#my_output = (next(train_generator))


# Keras CNN
################################################################################
import tensorflow as tf
tf.python.control_flow_ops = tf
# Initial Setup for Keras
from keras.models import Sequential
#from keras.layers.core import Dense, Activation, Flatten, Lambda, ELU
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D


# Preprocess incoming data, centered around zero with small standard deviation 
# Normalize and mean center
# model.add(Lambda(lambda x: x/127.5 - 1.,
#                  input_shape=(ch, row, col),
#                  output_shape=(ch, row, col)))

#ch, row, col = 3, 160, 320
ch, row, col = 3, 85, 320

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch = len(train_samples),
                    validation_data = validation_generator,
                    nb_val_samples = len(validation_samples),
                    nb_epoch=3)

model.save('model.h5')