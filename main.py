import numpy
import scipy.special
import matplotlib.pyplot

#neural network class definition
class neuralNetwork:

    #initialize the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #set number of nodes in each input, hidden, and output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #learning rate
        self.lr = learningrate

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        #weight matrices initialized with random values - normalization is better than random values between 0 and 1
        #self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5) # input to hidden layer weights
        #self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5) # hidden to output layer weights

        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes)) # input to hidden layer weights with normalization
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes)) # hidden to output layer weights with normalization

        #activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    #train the neural network
    def train(self, inputs_list, targets_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        #error is the (target - actual)
        output_errors = targets - final_outputs

        #hidden layer errors are the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        #update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    #query the neural network
    def query(self, inputs_list):

        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        #return the final outputs
        return final_outputs

#set number of nodes in each layer
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

#set learning rate
learning_rate = 0.3

#create an instance of the neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#load the mnist training data CSV file into a list
training_data_file = open("/u/kieranm/Documents/Dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#train the neural network

#epochs = 2  # number of times to go through the training data set
epochs = 4

#go through all records in the training data set
for e in range(epochs):
    for record in training_data_list:
        #split the record by commas
        all_values = record.split(',')
        #scale and shift the inputs
        inputs = (numpy.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
        #create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        #all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99

        #train the neural network with the inputs and targets
        n.train(inputs, targets)
        pass
    pass

#test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

#load the mnist test data CSV file into a list
test_data_file = open("/u/kieranm/Documents/Dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    # scale and shift the inputs
    inputs = (numpy.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print(label, "network's answer")
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print(scorecard_array.sum() / scorecard_array.size)


