#! /usr/bin/python
import pickle
import numpy as np
from PIL import Image

def vectorize_imgs(img_path):
    with Image.open(img_path) as img:
        arr_img = np.asarray(img, dtype='float32')
        return arr_img

def read_csv_file(csv_file):
    x, y = [], []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            path, label = line.strip().split()
            x.append(vectorize_imgs(path))
            y.append(int(label))
    return np.asarray(x, dtype='float32'), np.asarray(y, dtype='int32')

def read_csv_pair_file(csv_file):
    x1, x2, y = [], [], []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            p1, p2, label = line.strip().split()
            x1.append(vectorize_imgs(p1))
            x2.append(vectorize_imgs(p2))
            y.append(int(label))
    return np.asarray(x1, dtype='float32'), np.asarray(x2, dtype='float32'), np.asarray(y, dtype='int32')

def load_data():
    with open('data/dataset.pkl', 'rb') as f:
        testX1 = pickle.load(f)
        testX2 = pickle.load(f)
        testY  = pickle.load(f)
        validX = pickle.load(f)
        validY = pickle.load(f)
        trainX = pickle.load(f)
        trainY = pickle.load(f)
        return testX1, testX2, testY, validX, validY, trainX, trainY

if __name__ == '__main__':
    testX1, testX2, testY = read_csv_pair_file('data/test_set.csv')
    validX, validY = read_csv_file('data/valid_set.csv')
    trainX, trainY = read_csv_file('data/train_set.csv')

    print(testX1.shape, testX2.shape, testY.shape)
    print(validX.shape, validY.shape)
    print(trainX.shape, trainY.shape)

    with open('data/dataset.pkl', 'wb') as f:
        pickle.dump(testX1, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(testX2, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(testY , f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(validX, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(validY, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trainX, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trainY, f, pickle.HIGHEST_PROTOCOL)
