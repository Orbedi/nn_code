import gzip
# import cPickle
import _pickle as cPickle
import tensorflow as tf
import numpy as np

# https://www.apsl.net/blog/2017/12/05/tensor-flow-para-principiantes-i/
# https://www.tensorflow.org/versions/r1.1/get_started/mnist/pros
# https://ml4a.github.io/ml4a/es/looking_inside_neural_nets/
# http://naukas.com/2015/12/09/acertando-quinielas-redes-neuronales/
# http://ssalva.bitballoon.com/blog/2016-08-30-tensorflow/

# Para la convolution:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]

def porcentaje(todo):
    return (2*todo)/100

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set

val_x, val_y = valid_set

test_x, test_y = test_set
# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt


#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print train_y[57]

train_y = one_hot(train_y, 10)
val_y_one_hot = one_hot(val_y, 10)
test_y_one_hot = one_hot(test_y, 10)

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

# Buenos valores: 50 neuronas, batch_size = 10, 200 iteraciones, 0.5 porciento, 0.5 umbral error

W1 = tf.Variable(np.float32(np.random.rand(784, 50)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(50)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(50, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1) # Modelo lineal (logits): tf.matmul(x, W1) + b1
# h = tf.matmul(x, W1) + b1  # Try this! # Normalizamos los valores del modelo usando softmax y sigmoid
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

#loss = tf.reduce_sum(tf.square(y_ - y)) # Funcion de coste
# Funcion de entropia cruzada:
loss =  tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01  Metodo de optimizacion

init = tf.initialize_all_variables()

sess = tf.Session()  # Iniciamos la sesion
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 20
training_error = 100

graficaError = []
graficaErrorValidacion = []

for epoch in range(200):
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    training_error_old = training_error

    training_error = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    graficaError.append(training_error)

    validation_error = sess.run(loss, feed_dict={x: val_x, y_: val_y_one_hot})
    graficaErrorValidacion.append(validation_error)

    print("Epoch #:", epoch, "Error train: ", training_error)
    print("Epoch #:", epoch, "Error valid: ", validation_error)
    #result = sess.run(y, feed_dict={x: test_x})  # La 'y' es el modelo
    #for b, r in zip(test_y_one_hot, result):
    #    print(b, "-->", r)
    print("----------------------------------------------------------------------------------")

    if (abs(training_error_old - training_error) < porcentaje(training_error_old)):
        if (training_error < 0.5):
            break

# Precision
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_x, y_: one_hot(test_y, 10)}))

# GRAFICA
plt_train, = plt.plot(graficaError, label='Error entrenamiento')
plt_valid, = plt.plot(graficaErrorValidacion, label='Error validacion')
plt.legend(handles=[plt_train,plt_valid])
plt.xlabel("epoch")
plt.ylabel("error")
plt.show()

