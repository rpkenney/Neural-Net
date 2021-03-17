# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:56:25 2021

@author: robert
"""

import numpy as np

class NeuralNetwork:
    
    def __init__(self, input, expectedOutputs):
        #an array of all potential inputs that could go into the system
        self.input = input
        # an array of expected outputs, with the expected output for each input in the same index
        self.expectedOutputs = expectedOutputs
        #the weights of each input node, in this nn there is only one node in the hidden layer
        self.weights = [[np.random.uniform(0, 1)], [np.random.uniform(0, 1)], [np.random.uniform(0, 1)]]
    
    def sigmoid(self, weightedInput):
        return 1 / (1 + np.exp(-weightedInput))
    
    def derivativeSigmoid(self, weightedInput):
        return weightedInput * (1 - weightedInput)
        
    def feedForward(self):
        #gets the weighted output for the single node in the hidden layer using a dot product of all inputs and all weights
        self.hiddenLayerOutput = self.sigmoid(np.dot(self.input, self.weights))
        
    def backPropogation(self):
        #
        self.error = self.expectedOutputs - self.hiddenLayerOutput
        
        #change in cost with respect to hiddenLayerOutput(error is fixed, and the derivitve of the sigmoid is taken)
        costPrime = self.error * self.derivativeSigmoid(self.hiddenLayerOutput)
        
        #for future layers, the formula looks more like what is above dotted with the output weights of that layer, 
        #dotted with derivative sigmoid of that layers output, as per the chain rule
        
        #update the weights to reflect the cost
        self.weights += np.dot(self.input.T, costPrime)

    def training(self, iterations):
        for i in range(iterations):
            self.feedForward()
            self.backPropogation()
    
    def prediction(self, unseen_input):
        return self.sigmoid(np.dot(unseen_input, self.weights))

inputs = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]])

expectedOutputs = np.array([[0], [0], [0], [1], [1], [1]])

NN = NeuralNetwork(inputs, expectedOutputs)

NN.training(25000)

test = np.array([0, 1, 1])
print(NN.prediction(test))

