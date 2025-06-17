'''This neural network will be trained to recognize handwritten letters from the dataset.'''

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# load the training dataset
dataset = np.loadtxt('/u/kieranm/Documents/Dataset/letters.csv', delimiter=',')

#select a random subset of the dataset
np.random.seed(42)  # for reproducibility
np.random.shuffle(dataset)  # shuffle the dataset randomlyc
dataset = dataset[:5000]  # select the first 5000 samples for training

# split the dataset into input (X) and output (y) variables
X = dataset[:, 1:]  # input features (all columns except the first)
y = dataset[:, 0]   # output variable (first column, letters A-Z)

# Normalize input features to 0-1
X = X / 255.0

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert letters to one-hot encoding
num_classes = 26  # number of classes (A-Z)
y_train_one_hot = np.zeros((y_train.size, num_classes))
for i in range(y_train.size):
    y_train_one_hot[i, int(y_train[i]) - 1] = 1  # assuming letters are encoded as 1-26 for A-Z
y_test_one_hot = np.zeros((y_test.size, num_classes))
for i in range(y_test.size):
    y_test_one_hot[i, int(y_test[i]) - 1] = 1

# define the keras model
model = Sequential()  # a sequential model is a linear stack of layers
model.add(Dense(128, input_shape=(X.shape[1],), activation='relu'))  # add a dense layer with 128 neurons, input shape of X, and ReLU activation function
model.add(Dense(64, activation='relu'))  # add another dense layer with 64 neurons and ReLU activation
model.add(Dense(num_classes, activation='softmax'))  # add the output layer with num_classes neurons and softmax activation for multi-class classification

#plot a value from the dataset
# plt.imshow(X_train[4].reshape(28, 28), cmap='gray')  # reshape the first training sample to 28x28 for visualization
# plt.title(f'Label: {chr(int(y_train[4]) + 65)}')  # display the label as a letter (A-Z)
# plt.axis('off')  # turn off the axis
# plt.show()  # show the plot 

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # use categorical crossentropy loss function, Adam optimizer, and accuracy metric
# fit the model to the training data
model.fit(X_train, y_train_one_hot, epochs=60, batch_size=32, verbose=1)  # train the model for 20 epochs with a batch size of 32
# evaluate the model on the test data
_, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)  # evaluate the model on the test data
print('Test Accuracy: %.2f' % (accuracy * 100))  # print the accuracy as a percentage
# make predictions with the model
predictions = model.predict(X_test)  # predict the probabilities for the input data
# convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1) + 1  # get the index of the highest probability and add 1 to match the letter encoding (1-26)
# print the first 10 predicted classes
print('First 10 predicted classes:', predicted_classes[:10])  # print the first 10 predicted classes
# Print the corresponding letters for the first 10 predictions
predicted_letters = [chr(c + 65) for c in predicted_classes[:10]]
print('First 10 predicted letters:', predicted_letters)
# plot the first 10 images and their predicted labels
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {predicted_letters[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()