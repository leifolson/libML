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
            - layers: list of layer sizes from left to right, including the input layer size
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
        self.layers = layers
        self.weights = self._initWeights(layers)
        self.bestWeights = self.weights
        

    def train(self, epochs, learningRate):
        # TODO: implement
        
        for i in range(epochs):
            acts = self._forwardProp()
            
            # update weights
            self._backProp(acts, learningRate)
            
            print self.weights
            
            print 'targets: ' , self.targets
            print 'acts: ' , acts[len(self.layers) - 2]
            print 'errors: ' , sum((self.targets - acts[len(self.layers) - 2])**2)
            
            idx = range(self.data.shape[0])
            np.random.shuffle(idx)
            self.data = self.data[idx,:]
            self.targets = self.targets[idx, :]
            
        return 

    def predict(self):
        # TODO: implement
        return

    def _forwardProp(self):
        '''
        Forward propagates through the network, returning a dictionary
        of activations for each layer of the network.
        
        '''
        # forward propagate to get activations at each layer
        activations = {}
        
        numLayers = len(self.layers)
        
        # get activation function type for hidden layers
        h = self._getActivationFunc(self.hiddenType)
        
        # add bias to input
        trainEx = np.insert(self.data, 0, 1, axis = 1)
        
        # compute input to first layer
        neuralInput = np.dot(trainEx, self.weights[0].T)
        
        for i in range(numLayers - 1):
            # check if we need to get the output layer activation function type
            if(i == numLayers - 2):
                h = self._getActivationFunc(self.outType)
            
            # compute activations and store them
            acts = h(neuralInput)    
            activations.update({i : acts})

            # add bias node for next input and compute next input if needed
            if(i != numLayers - 2):
                acts = np.insert(acts, 0, 1, axis = 1)
                neuralInput = np.dot(acts, self.weights[i + 1].T)
        
        return activations
        

    def _backProp(self, activations, learningRate):
        '''
        Runs backpropagation for the Neural Network.
        Requires the activations dictionary for each layer.
        
        Also updates network weights
        '''
        
        numLayers = len(self.layers)

        # start at the output and propagate back
        deltas = {}
        
        # compute output errors
        actFuncDx = self._getActFuncDx(self.outType)
        outputLayer = numLayers - 2
        delta = (self.targets - activations[outputLayer]) * actFuncDx(activations[outputLayer]) 
        deltas.update({outputLayer : delta})
        
        # compute the hidden layer errors
        for i in reversed(range(numLayers - 2)):
            actFuncDx = self._getActFuncDx(self.hiddenType)
            weights = self.weights[i+1]
            
            # remember, we dont compute derivatives on bias nodes so we ignore those weights for now
            delta = actFuncDx(activations[i]) * (np.dot(deltas[i+1], weights[:,1:]))
            deltas.update({i : delta})
            
            
        # update the weights
        self._updateWeights(activations, deltas, learningRate)
        
        return
    
    
    def _updateWeights(self, activations, deltas, learnRate):
        # TODO: Implement
     
        # update input weights
        inputUpdate = np.zeros(self.weights[0].shape)
        inputWithBias = np.insert(self.data, 0, 1, axis = 1)
        inputUpdate = learnRate * np.dot(deltas[0].T, inputWithBias)
        
        self.weights[0] += inputUpdate
        
        # now update the other weights
        for i in range(len(self.layers) - 2):
            act = activations[i]
            act = np.insert(act, 0, 1, axis = 1)

            update = learnRate * np.dot(deltas[i+1].T, act)
            self.weights[i + 1] += update
        
        return


    def _initWeights(self, layers):
        '''
        Initializes a weight dictionary that stores the weight matrix
        at each layer of the NN.  Weights are organized into rows for
        each of the neurons.
        
        Following Marsland (2009), the weights are initialized to very
        small positive and negative pseudorandom values
        '''

        # weights will be stored in a dict
        weights = {}       
         
        # generate weights
        for i in range(len(layers) - 1):
            layerWeights = np.random.rand(layers[i + 1], layers[i] + 1) * 0.1 - 0.05
            weights.update({i : layerWeights})
            
        return weights
    
    def _getActivationFunc(self, type):
        if type == 'linear':
            def f(x):
                return x
            
        elif type == 'logistic':
            def f(x):
                return 1 / (1 + np.exp(-x))
            
        elif type == 'tanh':
            def f(x):
                return np.tanh(x)
            
        return f
    
    
    def _getActFuncDx(self, type):
        if type == 'linear':
            def f(x):
                return 1
            
        elif type == 'logistic':
            def f(x):
                return x * (1 - x)
            
        elif type == 'tanh':
            def f(x):
                return (1 - np.tanh(x) ** 2)
            
        return f