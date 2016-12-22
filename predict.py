#! /usr/bin/python
import pickle
from deepid1 import *
import tensorflow as tf
from scipy.spatial.distance import cosine, euclidean

if __name__ == '__main__':
    with tf.Session() as sess:
        saver.restore(sess, 'checkpoint/30000.ckpt')
        h1 = sess.run(h5, {h0: testX1})
        h2 = sess.run(h5, {h0: testX2})

    pre_y = np.array([cosine(x, y) for x, y in zip(h1, h2)])
    
    def part_mean(x, mask):
        z = x * mask
        return float(np.sum(z) / np.count_nonzero(z))
    
    true_mean = part_mean(pre_y, testY)
    false_mean = part_mean(pre_y, 1-testY)
    print(true_mean, false_mean)
    
    print(np.mean((pre_y < (true_mean + false_mean)/2) == testY.astype(bool)))
