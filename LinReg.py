'''
Author: Clinton Olson
Contact: clint.olson2@gmail.com

LinReg implements simple multivariate linear regression
'''
import numpy as np

class LinReg:

    def __init__(self):
        self.data = None
        self.targets = None
        self.nsamps = 0
        self.nvars = 0
        self.weights = None
        

    def train(self, data, targets, eta = 0.2, lamb = 0, useNormEq = True):
        '''
        Inputs:
            - data: samples in rows, vars in columns
            - targets: array of target values
            - eta: learning rate, default = 0.2
            - lamb: regularzation term, default = 0  i.e., no regularization
            - useNormEq: flag for using the normal equation to solve directly
                         defaults to True
        '''
        self.data = data
        self.targets = targets
        self.nsamps, self.nvars = data.shape

        if useNormEq:
            # try to invert but revert to pseudo-inverse if necessary
            try:
                xInv = np.linalg.inv(np.dot(data.T,data))

            except LinAlgError:
                xInv = np.linalg.pinv(np.dot(data.T,data))

            self.weights = np.dot(np.dot(xInv, data.T),targets.T)
            
        else:
            # do the gradient descent thing
            x = 1 


        # initialize weights to small positive and negative values
        #self.weights = np.random.rand(self.nvars) * -0.1 - 0.05
