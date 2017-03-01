import os
import struct
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

def read_mnist(dataset = "training", path = "./data/"):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
        flbl.close()
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
        fimg.close()
    return img, lbl, num, rows, cols

def read_test(file_name = "test-image", path = "./data/"):
    fname_img = os.path.join(path, file_name)
    with open(fname_img, 'rb') as fimg:
        mmagic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows, cols)
        fimg.close()
    return img, num

X_train, y_train, num_train, rows, cols = read_mnist()
X_test, num_test = read_test()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(num_train, 1, rows, cols)
    X_test = X_test.reshape(num_test, 1, rows, cols)
    input_shape = (1, rows, cols)
else:
    X_train = X_train.reshape(num_train, rows, cols, 1)
    X_test = X_test.reshape(num_test, rows, cols, 1)
    input_shape = (rows, cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, 10)

# parameters
nb_filters = 64
pool_size = (2, 2)
kernel_size = (3, 3)
batch_size = 128
nb_epoch = 15

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],border_mode='valid',input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
classes = model.predict_classes(X_test, batch_size=10000)

with open('./data/result.csv','w') as out_file:
    out_file.write('id,label\n')
    for i in range(len(classes)):
        out_file.write(str(i)+','+str(classes[i])+'\n')
    out_file.close()
