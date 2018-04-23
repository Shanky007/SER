import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool, max_pool_1d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.layers.recurrent import lstm
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import numpy as np


def concordance_cc(prediction, ground_truth):
    """Defines concordance loss for training the model.

    Args:
       prediction: prediction of the model.
       ground_truth: ground truth values.
    Returns:
       The concordance value.
    """
    print(np.shape(prediction))
    print(np.shape(ground_truth))
    pred_mean, pred_var = tf.nn.moments(prediction, (0,))
    gt_mean, gt_var = tf.nn.moments(ground_truth, (0,))

    mean_cent_prod = tf.reduce_mean((prediction - pred_mean) * (ground_truth - gt_mean))

    return 1 - (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))



# Smaller 'AlexNet'
# https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
print('[+] Building CNN')

trainX = np.load('train_input.npy')
trainY = np.load('train_output.npy')
evalX = np.load('valid_input.npy')
evalY = np.load('valid_output.npy')
testX = np.load('test_input.npy')
testY = np.load('test_output.npy')

trainX = np.concatenate((trainX, evalX[:60000]))
trainY = np.concatenate((trainY, evalY[:60000]))
trainX  = np.concatenate((trainX, testX[:30000]))
trainY = np.concatenate((trainY, testY[:30000]))

evalX = evalX[60000:]
evalY = evalY[60000:]
testX = testX[30000:]
testY = testY[30000:]

print(np.shape(trainX))
print(np.shape(trainY))
print(np.shape(testX))
print(np.shape(testY))
print(np.shape(evalX))
print(np.shape(evalY))

# Building convolutional network
network = input_data(shape=[None, 640], name='input')

network = tf.expand_dims(network, 2)
# network = conv_1d(network, 512, 3, padding='valid', activation='relu', regularizer="L2")
# network = max_pool_1d(network, 2)
# network = conv_1d(network, 512, 5, padding='valid', activation='relu', regularizer="L2")
# network = max_pool_1d(network, 2)
branch1 = conv_1d(network, 128, 1, padding='valid', activation='relu', regularizer="L2")
branch2 = conv_1d(network, 128, 2, padding='valid', activation='relu', regularizer="L2")
branch3 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
branch4 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
branch5 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
# branch6 = conv_1d(network, 128, 6, padding='valid', activation='relu', regularizer="L2")
# branch7 = conv_1d(network, 128, 7, padding='valid', activation='relu', regularizer="L2")
# branch8 = conv_1d(network, 128, 8, padding='valid', activation='relu', regularizer="L2")
network = merge([branch1, branch2, branch3, branch4, branch5], mode='concat', axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)

# network = fully_connected(network, 64, activation='tanh')
# network = dropout(network, 0.8)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.8)
network = tf.expand_dims(network, 2)
network = lstm(network, 128, return_seq=True)
network = lstm(network, 128)
network = fully_connected(network, 64, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='tanh')
network = regression(network, optimizer='adam', learning_rate=0.001, loss=concordance_cc, name='target')
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
# model = tflearn.DNN(network, tensorboard_verbose=0)
model.load('./saved_model/cnn.tfl')
# model.fit(trainX, trainY, n_epoch=1, shuffle=True, validation_set=(evalX, evalY), show_metric=True, batch_size=128)
# model.save('./saved_model/cnn.tfl')

for i,x in enumerate(testX[1000:]):
    pred = model.predict([x])
    print("Prediction:", pred[0])
    print("Actual    :", testY[i])
    print("========================================")

