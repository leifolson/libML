'''
Author: Clinton Olson
Email: clint.olson2@gmail.com

Implements a general Feed-Forward Neural Network

The goal is to be able to specify any number of hidden layers
'''

import numpy as np

class FFNN:
    
    def __init__(self, data, targets, layers, hiddenType = 'logistic', outType = 'logistic'):
        '''
        Inputs:
            - data: numpy array of training data, samples in rows
            - layers: list of layer sizes from left to right
                      do not include input layer
            - targets: numpy array of target values
            - hiddenType: type of hidden layer neurons
                          possible values of linear or logistic currently
            - outType: type of output layer neurons
        '''
        self.data = data
        self.targets = targets
        self.hiddenType = hiddenType
        self.outType = outType
        self.numEx, self.numVars = data.shape
        self.weights = self._initWeights(layers)
        

    def train(self):
        # TODO: implement
        return 

    def predict(self):
        # TODO: implement
        return

    def _forwardProp(self):
        # TODO: implement
        return


    def _backProp(self):
        # TODO: implement
        return


    def _initWeights(self, layers):
        # TODO: implement
        # initialize weights to small pos/neg random weights for
        # each of the layers in the network

        # weights will be stored in a dict
        weights = {}

        # first generate input weights
        wIn = np.random.rand(layers[0], self.numVars+1) * 0.1 - 0.05
        weights.update({0:wIn})        
         
        # now generate hidden weights
        #for i in range(len(layers)):
            
            
            
        return weights
    
    def _getActivationFunc(self, type):
        if type == 'linear':
            def f(x):
                return x
            
        elif type == 'logistic':
            def f(x):
                return 1 / (1 + np.exp(-x))
            
        return f