import os
import struct
import numpy as np
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
tf.set_random_seed(1337) # for reproducibility
from tensorflow.contrib import learn

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

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def next_batch(X_train, y_train, current_index, batch_size=128):
    start = current_index
    current_index += batch_size
    if current_index > X_train.shape[0]:
        perm = np.arange(X_train.shape[0])
        np.random.shuffle(perm)
        X_train = X_train[perm]
        y_train = y_train[perm]
        start = 0
        current_index = batch_size
    X_batch = X_train[start:current_index]
    y_batch = y_train[start:current_index]
    return X_batch, y_batch, current_index, X_train, y_train

X_train, y_train, num_train, rows, cols = read_mnist()
X_test, num_test = read_test()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = dense_to_one_hot(y_train)

X_train = np.reshape(X_train,[-1, rows, cols, 1])
X_test = np.reshape(X_test,[-1, rows, cols, 1])
#y_train = np.reshape(y_train,[-1, 1])

# input and output
X = tf.placeholder(tf.float32, shape=[None, rows, cols, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# parameters
nb_filters = 64
pool_size = (2, 2)
kernel_size = (3, 3)
batch_size = 128
nb_epoch = 45

# model structure
W_conv1 = weight_variable([kernel_size[0], kernel_size[1], 1, nb_filters])
b_conv1 = bias_variable([nb_filters])
h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([kernel_size[0], kernel_size[1], nb_filters, nb_filters])
b_conv2 = bias_variable([nb_filters])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

keep_prob1 = tf.placeholder(tf.float32)
h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob1)

flat_rows = int(rows/4)
flat_cols = int(cols/4)

W_fc1 = weight_variable([flat_rows * flat_cols * nb_filters, 512])
b_fc1 = bias_variable([512])
flat = tf.reshape(h_pool2_drop, [-1, flat_rows * flat_cols * nb_filters])
h_fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)

keep_prob2 = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob2)

W_fc2 = weight_variable([512, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

total_batch = int(X_train.shape[0]/batch_size)
current_index = 0
train_accuracy = 0
iteration = total_batch * nb_epoch

decoded = tf.argmax(y_conv, axis=1)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(iteration+1):
        X_batch, y_batch, current_index, X_train, y_train = next_batch(X_train, y_train, current_index, batch_size)
        if i % total_batch == 0:
            train_accuracy /= total_batch
            print("epoch %d/%d, training accuracy %g"%(i/total_batch, nb_epoch, train_accuracy))
            train_accuracy = 0
            if i == iteration:
                break
        train_accuracy += accuracy.eval(feed_dict={X: X_batch, y_: y_batch, keep_prob1: 1.0, keep_prob2: 1.0}) 
        _, loss_val = sess.run([train_step, loss],feed_dict={X: X_batch, y_: y_batch, keep_prob1: 0.25, keep_prob2: 0.5})
    predicted = sess.run(decoded,feed_dict={X: X_test, keep_prob1: 1.0, keep_prob2: 1.0})
    with open('./data/result_tf.csv','w') as out_file:
        out_file.write('id,label\n')
        for i in range(len(predicted)):
            out_file.write(str(i)+','+str(predicted[i])+'\n')
        out_file.close()
