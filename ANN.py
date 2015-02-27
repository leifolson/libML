'''
Author: Clinton Olson
Email: clint.olson2@gmail.com

Implements an Artificial Neural Network Class

The goal is to be able to specify any number of hidden layers
'''
import numpy as np

class ANN:

    def __init__(self, trainData, targets, layers):
        '''
        Inputs:
            - trainData: numpy array of training data, samples in rows
            - layers: list of hidden layer sizes in order from left to right
            - targets: numpy array of target values

        '''
        self.data = trainData
        self.targets = targets
        self.nSamps, self.nVars = trainData.shape
        self.weights = self._initWeights(layers)
        

    def train(self):
        # TODO: implement
        return 


    def forwardProp(self):
        # TODO: implement
        return


    def backProp(self):
        # TODO: implement
        return


    def _initWeights(self, layers):
        # TODO: implement
        # initialize weights to small pos/neg random weights for
        # each of the layers in the network

        # weights will be stored in a dict
        weights = {}

        # first generate input weights
        wIn = np.random.rand(layers[0], self.nVars+1) * -0.1 - 0.05
        weights.update({0:wIn})        
         
        # now generate hidden weights
        #for i in range(len(layers)):
            
            
            
        return weights 
        
