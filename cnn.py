import numpy as np
np.random.seed(111)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from matplotlib import pyplot as plt

# constants
batch_size = 128
NUM_CLASSES = 10
epochs = 12
IMG_WIDTH, IMG_HEIGHT = 28, 28

# split data on train and test datasets
(x_train_origin, y_train_origin), (x_test, y_test) = mnist.load_data()

x_train = x_train_origin[:1000]
y_train = y_train_origin[:1000]

# prepare dataset item shapes
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, IMG_HEIGHT, IMG_WIDTH)
    x_test = x_test.reshape(x_test.shape[0], 1, IMG_HEIGHT, IMG_WIDTH)
    input_shape = (1, IMG_HEIGHT, IMG_WIDTH)
else:
    x_train = x_train.reshape(x_train.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)


# normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert to categorical classification
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)


# model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# tensorboard callback
tb_callback = keras.callbacks.TensorBoard(log_dir='./logs')

# training
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.25,
          callbacks=[tb_callback])

# estimate
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# show image
plt.imshow(x_train_origin[0], cmap='gray')
plt.show()

predicted = model.predict(np.expand_dims(x_train[0], axis=0))
print(np.argmax(predicted[0]))
