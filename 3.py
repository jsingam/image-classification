#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Conv2DTranspose, UpSampling2D
import numpy as np
from keras.constraints import max_norm
from sklearn.model_selection import train_test_split


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


noise_factor = 0.25
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0. , 1.)
x_test_noisy = np.clip(x_test_noisy, 0. , 1.)


# In[4]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train_noisy = x_train_noisy.reshape(x_train_noisy.shape[0], 28, 28, 1)
x_test_noisy = x_test_noisy.reshape(x_test_noisy.shape[0], 28, 28, 1)


# In[5]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train_noisy = x_train_noisy.astype('float32')
x_test_noisy = x_test_noisy.astype('float32')


# In[6]:


x_train /= 255
x_test /= 255
x_train_noisy /= 255
x_test_noisy /= 255


# In[7]:


# Convolutional Autoencoder Neural Network

# Encoding the image
auto_encoder = Sequential()
auto_encoder.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28,28,1)))
auto_encoder.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
auto_encoder.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
auto_encoder.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Reconstructing the image
auto_encoder.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
auto_encoder.add(UpSampling2D((2, 2)))
auto_encoder.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
auto_encoder.add(UpSampling2D((2, 2)))
auto_encoder.add(Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'))


# In[12]:


auto_encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
auto_encoder.fit(x_train_noisy, x_train,epochs=5)


# In[8]:


auto_encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
auto_encoder.fit(x_train_noisy, x_train,epochs=5)


# In[9]:


# Removing the noice through auto encoding of the noisy data

x_train_noice_reduced = auto_encoder.predict(x_train_noisy)
x_test_noice_reduced = auto_encoder.predict(x_test_noisy)


# In[12]:


model = Sequential()
model.add(Conv2D(28, activation='relu', kernel_size=(3,3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)


# In[13]:


model.evaluate(x_test, y_test)


# In[ ]:




