from numpy import loadtxt #for loading data
from keras.models import Sequential # for creating a model
from keras.layers import Dense # for adding layers to the model

# load the training dataset
dataset = loadtxt('/u/kieranm/Documents/Dataset/pima-indians-diabetes.data.csv', delimiter=',')
# split the dataset into input (X) and output (y) variables
X = dataset[:,0:8]  # input features (first 8 columns)
y = dataset[:,8]   # output variable (last column)

# define the keras model
'''This code creates a simple feedforward neural network using Keras. 
We use a ReLU activation function for the hidden layers and a sigmoid activation function for the output layer, which is suitable for binary classification tasks.
The model consists of two hidden layers with 12 and 8 neurons respectively, and an output layer with 1 neuron.'''
model = Sequential() # a sequential model is a linear stack of layers, each layer has weights that are updated during training
model.add(Dense(12, input_shape=(8,), activation='relu')) # add a dense layer with 12 neurons, input shape of 8 (features), and ReLU activation function
model.add(Dense(8, activation='relu')) # add another dense layer with 8 neurons and ReLU activation
model.add(Dense(1, activation='sigmoid')) # add the output layer with 1 neuron and sigmoid activation for binary classification

# compile the model
'''
Binary_crossentropy is used for binary classification problems, it measures the performance of a model whose output is a probability value between 0 and 1
The adam optimizer is an adaptive learning rate optimization algorithm that is popular for training deep learning models
The accuracy metric is used to evaluate the performance of the model during training and testing
'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # use binary crossentropy loss function, Adam optimizer, and accuracy metric

# fit the model to the training data
model.fit(X, y, epochs=150, batch_size=10) # train the model for 150 epochs with a batch size of 10

# evaluate the model on the training data
_, accuracy = model.evaluate(X, y) # evaluate the model on the training data
# the above line returns the loss and accuracy of the model on the training data
print('Accuracy: %.2f' % (accuracy * 100)) # print the accuracy as a percentage

