# (C) Aparajita Haldar
#
# Program using EasyTensorFlow package
# to evolve good neural network structure
# for digit recognition classification problem. 
#

import metadata, tf_functions, evolve_functions
import numpy

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
training_data = load_digits()

print("Matrix 1797 images 64 pixels each ", training_data.data.shape)

trX = training_data.data
trY = training_data.target.reshape((1797, 1))

#net_type, opt, m = evolve_functions.evolve('classification', 'accuracy', trX, trY, max_layers=4, train_iters=10)
#print net_type, opt, m

X_train, X_test, y_train, y_test = train_test_split(trX, trY, test_size=0.33, random_state=42)

# using result obtained from evolve function
net_type = ['dropout', 1000, 'bias_add', 100, 'relu', 0, 'relu', 0, 'softmax']

reg = tf_functions.Classifier(net_type)

# training_steps = train_iters required
training_steps = 10

reg.train(X_train, y_train, training_steps)

p = reg.predict(X_test)
print p

