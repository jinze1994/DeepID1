#! /usr/bin/python
import pickle
import numpy as np
import tensorflow as tf
from vec import load_data

testX1, testX2, testY, validX, validY, trainX, trainY = load_data()
class_num = np.max(trainY) + 1

def weight_variable(shape):
    with tf.name_scope('weights'):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    with tf.name_scope('biases'):
        return tf.Variable(tf.zeros(shape))

def Wx_plus_b(weights, x, biases):
    with tf.name_scope('Wx_plus_b'):
        return tf.matmul(x, weights) + biases

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        weights = weight_variable([input_dim, output_dim])
        biases = bias_variable([output_dim])
        preactivate = Wx_plus_b(weights, input_tensor, biases)
        if act != None:
            activations = act(preactivate, name='activation')
            return activations
        else:
            return preactivate

def conv_pool_layer(x, w_shape, b_shape, layer_name, act=tf.nn.relu, only_conv=False):
    with tf.name_scope(layer_name):
        W = weight_variable(w_shape)
        b = bias_variable(b_shape)
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name='conv2d')
        h = conv + b
        relu = act(h, name='relu')
        if only_conv == True:
            return relu
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max-pooling')
        return pool

def accuracy(y_estimate, y_real):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):  
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        with tf.name_scope('accuracy'):  
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)  
        return accuracy

def train_step(loss):
    with tf.name_scope('train'):
        return tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.name_scope('input'):
    h0 = tf.placeholder(tf.float32, [None, 55, 47, 3], name='x')
    y_ = tf.placeholder(tf.float32, [None, class_num], name='y')

h1 = conv_pool_layer(h0, [4, 4, 3, 20], [20], 'Conv_layer_1')
h2 = conv_pool_layer(h1, [3, 3, 20, 40], [40], 'Conv_layer_2')
h3 = conv_pool_layer(h2, [3, 3, 40, 60], [60], 'Conv_layer_3')
h4 = conv_pool_layer(h3, [2, 2, 60, 80], [80], 'Conv_layer_4', only_conv=True)

with tf.name_scope('DeepID1'):
    h3r = tf.reshape(h3, [-1, 5*4*60])
    h4r = tf.reshape(h4, [-1, 4*3*80])
    W1 = weight_variable([5*4*60, 160])
    W2 = weight_variable([4*3*80, 160])
    b = bias_variable([160])
    h = tf.matmul(h3r, W1) + tf.matmul(h4r, W2) + b
    h5 = tf.nn.relu(h)

with tf.name_scope('loss'):
    y = nn_layer(h5, 160, class_num, 'nn_layer', act=None)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    tf.scalar_summary('loss', loss)  

accuracy = accuracy(y, y_)
train_step = train_step(loss)

merged = tf.merge_all_summaries()  
saver = tf.train.Saver()

if __name__ == '__main__':
    def get_batch(data_x, data_y, start):
        end = (start + 1024) % data_x.shape[0]
        if start < end:
            return data_x[start:end], data_y[start:end], end
        return np.vstack([data_x[start:], data_x[:end]]), np.vstack([data_y[start:], data_y[:end]]), end
    
    data_x = trainX
    data_y = (np.arange(class_num) == trainY[:,None]).astype(np.float32)
    validY = (np.arange(class_num) == validY[:,None]).astype(np.float32)
    
    logdir = 'log'
    if tf.gfile.Exists(logdir):
        tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    train_writer = tf.train.SummaryWriter(logdir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(logdir + '/test', sess.graph)
    
    idx = 0
    for i in range(50001):
        batch_x, batch_y, idx = get_batch(data_x, data_y, idx)
        summary, _ = sess.run([merged, train_step], {h0: batch_x, y_: batch_y})
        train_writer.add_summary(summary, i)
    
        if i % 100 == 0:
            summary, accu = sess.run([merged, accuracy], {h0: validX, y_: validY})
            test_writer.add_summary(summary, i)
        if i % 5000 == 0 and i != 0:
            saver.save(sess, 'checkpoint/%05d.ckpt' % i)
