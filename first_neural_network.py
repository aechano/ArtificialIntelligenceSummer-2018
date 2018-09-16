#   Alexis Echano 2018
#   Using tutorial from machinelearningmastery.com and Diabetes database
#   Date: 9/16/2018, 2:50pm

#imports
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

'''--PART ONE: LOADING THE DATA--'''
#fix random seed for reproducibility
np.random.seed(7)

#this function loads the pima indians dataset
#delimiter acts like split
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

#splits data in to input (x) and output (y) variables
X = dataset[:,0:8]
Y = dataset[:,8]


'''

--PART TWO: DEFINING THE MODEL (SEQUENTIAL MODEL ADDING ONE LAYER AT A TIME)--

MAKE SURE the model has the right number of inputs

In our case, we will use the input_dim arg and setting it to 8 for the
8 input variables above!

This model is a fully-connected network structure with 3 layers

Fully connected layers are defined using the Dense class. 
We can specify the number of neurons in the layer as the first argument, the initialization method as the second 
argument as init and specify the activation function using the activation argument.

In this case, we initialize the network weights to a small random number generated from a uniform 
distribution (‘uniform‘), in this case between 0 and 0.05 because that is the default uniform weight initialization in 
Keras. Another traditional alternative would be ‘normal’ for small random numbers generated from a Gaussian distribution.

We will use the rectifier (‘relu‘) activation function on the first two layers and the sigmoid function in the output 
layer. It used to be the case that sigmoid and tanh activation functions were preferred for all layers. These days, 
better performance is achieved using the rectifier activation function. We use a sigmoid on the output layer to ensure 
our network output is between 0 and 1 and easy to map to either a probability of class 1 or snap to a hard 
classification of either class with a default threshold of 0.5.

We can piece it all together by adding each layer. The first layer has 12 neurons and expects 8 input variables. 
The second hidden layer has 8 neurons and finally, the output layer has 1 neuron to predict 
the class (onset of diabetes or not).

'''

#creating the actual model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

'''

--PART THREE: COMPILING THE MODEL--

Uses the backend now (Tensorflow) and automatically chooses the best way to represent the 
network for training and making predictions. When we compile, we need to give the model
some specifications such as the loss function, an optimizer and any metrics.

Loss: Logarithmic (binary_crossentropy)
Gradient Descent (decrease in loss in multiply dimensions): adam optimization
Type of AI: Classification, accuracy will be the metric if the model has done the task

'''

#compiling the model
model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

'''

--PART FOUR: FITTING THE MODEL--

This section is finally using the data to train the model by using fit().
Epochs are iterations the model goes thru when identifyinf loss/cost (epcochs)
Batches are number of instances until the model updates the weights (batch_size)

'''

#fitting the model using relatively small iterations and bacth size
model.fit(X, Y, epochs=150, batch_size=10)

'''

--PART FIVE: EVALUATING THE MODEL--

This is for testing new data but in our case we will just test
using old data!

Evaluate model using evaluate() and passing input and output

'''

#evaluating the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))