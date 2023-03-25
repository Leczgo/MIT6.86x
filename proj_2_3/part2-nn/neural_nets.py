import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    return max(0,x)

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    return 0 if x <= 0 else 1

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS (Initialized to floats instead of ints)
        self.input_to_hidden_weights = np.matrix('1. 1.; 1. 1.; 1. 1.')
        self.hidden_to_output_weights = np.matrix('1. 1. 1.') # (1 by 3 matrix)
        self.biases = np.matrix('0.; 0.; 0.') #(3 by 1 matrix)
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]

    def train(self, x1, x2, y):

        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = np.dot(self.input_to_hidden_weights,input_values) + self.biases  #(3 by 1 matrix)
        hidden_layer_activation = np.vectorize(rectified_linear_unit)(hidden_layer_weighted_input)
        output =  np.dot(self.hidden_to_output_weights,hidden_layer_activation)
        activated_output = np.vectorize(output_layer_activation)(output)

        ### Backpropagation ###

        # Compute gradients
        cost = 0.5 * (y - activated_output[0,0])**2
        output_layer_error = output_layer_activation_derivative(output[0,0]) * -1 * (y - activated_output[0,0])
        hidden_layer_derivative = np.vectorize(rectified_linear_unit_derivative)(hidden_layer_weighted_input)
        hidden_layer_error = np.multiply(self.hidden_to_output_weights.T,hidden_layer_derivative)  * output_layer_error #(3 by 1 matrix)

        #bias_gradients = output_layer_activation_derivative(self.biases)
        #bias_gradients = 1 * (np.vectorize(rectified_linear_unit_derivative)(hidden_layer_activation)) * self.hidden_to_output_weights * output_layer_activation_derivative(activated_output[0,0]) * output_layer_error
        bias_gradients = hidden_layer_error
        #bias_gradients = 1 * np.multiply(self.hidden_to_output_weights.T,(np.vectorize(rectified_linear_unit_derivative)(hidden_layer_activation))) * output_layer_error * output_layer_activation_derivative(activated_output[0,0])
        #hidden_to_output_weight_gradients = np.vectorize(rectified_linear_unit_derivative)(self.hidden_to_output_weights)
        #hidden_to_output_weight_gradients = hidden_layer_activation * output_layer_activation_derivative(output[0,0]) * output_layer_error
        #input_to_hidden_weight_gradients = np.vectorize(output_layer_activation_derivative)(self.input_to_hidden_weights)
        input_to_hidden_weight_gradients = np.dot(hidden_layer_error,input_values.T)
        hidden_to_output_weight_gradients = hidden_layer_activation * output_layer_error

        # Use gradients to adjust weights and biases using gradient descent
        new_biases = self.biases - (self.learning_rate * bias_gradients)
        new_input_to_hidden_weights = self.input_to_hidden_weights - (self.learning_rate * input_to_hidden_weight_gradients)
        new_hidden_to_output_weights = self.hidden_to_output_weights - (self.learning_rate * hidden_to_output_weight_gradients.T)
        
        self.biases = new_biases
        self.input_to_hidden_weights = new_input_to_hidden_weights
        self.hidden_to_output_weights = new_hidden_to_output_weights

        #print results
        #print("Hidden Layer Weighted Input: " + str(hidden_layer_weighted_input) + str(hidden_layer_weighted_input.shape) + str(type(hidden_layer_weighted_input)))
        #print("Hidden to Output Weights: " +  str(self.hidden_to_output_weights) + str(self.hidden_to_output_weights.shape) + str(type(self.hidden_to_output_weights)))
        #print("Hidden Layer Activation: " + str(hidden_layer_activation) + str(hidden_layer_activation.shape) + str(type(hidden_layer_activation)))
        #print("Output: " + str(output) + str(output.shape) + str( type(output) ))
        #print("Activated Output: " + str(activated_output) + str( activated_output.shape) + str(type(activated_output)))
        #print("Hidden Layer Derivative: " + str(hidden_layer_derivative) + str(hidden_layer_derivative.shape) + str(type(hidden_layer_derivative)))
        #print("Hidden Layer Error = " + str(hidden_layer_error) + str(hidden_layer_error.shape) + str(type(hidden_layer_error)))
        #print("Output Error = " + str(output_layer_error) + str(type(output_layer_error)))
        print("Input to Hidden Gradients: " + str(input_to_hidden_weight_gradients) + str(input_to_hidden_weight_gradients.shape) + str(type(input_to_hidden_weight_gradients)))
        print("Bias Gradients: " + str(bias_gradients) + str(bias_gradients.shape) + str(type(bias_gradients)))
        print("Hidden to Output Gradients: " + str(hidden_to_output_weight_gradients) + str(hidden_to_output_weight_gradients.shape) + str(type(input_to_hidden_weight_gradients)))
        print("Updated Input to Hidden Weights: " + str(new_input_to_hidden_weights) + str(new_input_to_hidden_weights.shape) + str(type(new_input_to_hidden_weights)))
        print("Updated Biases: " + str(new_biases) + str(new_biases.shape) + str(type(new_biases)))
        print("Updated Hidden to Output Weights: " + str(new_hidden_to_output_weights) + str(new_hidden_to_output_weights.shape) + str(type(new_hidden_to_output_weights)))

    def predict(self, x1, x2):

        input_values = np.matrix([[x1],[x2]])

        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input = np.dot(self.input_to_hidden_weights,input_values) + self.biases
        hidden_layer_activation = np.vectorize(rectified_linear_unit)(hidden_layer_weighted_input)
        output = np.dot(self.hidden_to_output_weights,hidden_layer_activation)
        activated_output = np.vectorize(output_layer_activation)(output)

        return activated_output.item()

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x[0], x[1], y)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return

NeuralNetwork().train(-7,-1,-8)

x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
x.test_neural_network()
