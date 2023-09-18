import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Required Libraries
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import TensorBoard
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2
import os
# Your Project Path
os.chdir("E:/Users/İndirilenler/yapay_zeka_proje_Lane_Detection/")
# Training Data
train_images = pickle.load(open("full_CNN_train.p", "rb" ))

# Processed Images
labels = pickle.load(open("full_CNN_labels.p", "rb" ))


# Converts Data Into an Array
train_images = np.array(train_images)
labels = np.array(labels)

# Normalize labels - training datas get normalized to start in the network
labels = labels / 255

# Images are shuffled and separated into training and validation
train_images, labels = shuffle(train_images, labels)

X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)


# The number of samples given to the network, the number of repetitions and the pool size determine the size of the pool to be created.
batch_size = 256
epochs = 10
pool_size = (2, 2)
input_shape = X_train.shape[1:]


# Sinir ağı
model = Sequential()
# Gelen verileri normalize eder ve işlemin başlaması için ilk katmana giriş değeri atarız
model.add(BatchNormalization(input_shape=input_shape))

# Convolution 1
model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))

# Convolution 2
model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))

# Pooling 1
model.add(MaxPooling2D(pool_size=pool_size))

# Convolution 3
model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))

# Convolution 4
model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
model.add(Dropout(0.2))

# Convolution 5
model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
model.add(Dropout(0.2))

# Pooling 2
model.add(MaxPooling2D(pool_size=pool_size))

# Convolution 6
model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
model.add(Dropout(0.2))

# Convolution 7
model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
model.add(Dropout(0.2))

# Pooling 3
model.add(MaxPooling2D(pool_size=pool_size))

# Upsample 1 
#Verilerin satırlarını ve sütunlarını tekrarlar
model.add(UpSampling2D(size=pool_size))

# Deconvolution 1
model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'))
model.add(Dropout(0.2))

# Deconvolution 2
model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'))
model.add(Dropout(0.2))

# Upsample 2
model.add(UpSampling2D(size=pool_size))

# Deconvolution 3
model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'))
model.add(Dropout(0.2))

# Deconvolution 4
model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4'))
model.add(Dropout(0.2))

# Deconvolution 5
model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5'))
model.add(Dropout(0.2))

# Upsample 3
model.add(UpSampling2D(size=pool_size))

# Deconvolution 6
model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6'))

# Final
model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final'))

# Sinir ağı


from keras import optimizers

datagen = ImageDataGenerator(channel_shift_range=0.2,horizontal_flip=True)
datagen.fit(X_train)

model.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,write_graph=True, write_images=True)

model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
steps_per_epoch = len(X_train)/(batch_size),
epochs=epochs, verbose=1, validation_data=(X_val, y_val))
# model.summary()
# print(len(model.layers)) 


# modelin eğtilmesi durur
model.trainable = False
model.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])
# modeli kaydet
model.save('full_CNN_model.h5')



