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
        

    def train(self, data, targets, eta = 0.2, lamb = 0, useNormEq = True, iters = 100):
        '''
        Inputs:
            - data: samples in rows, vars in columns
            - targets: array of target values
            - eta: learning rate, default = 0.2
            - lamb: regularzation term, default = 0  i.e., no regularization
            - useNormEq: flag for using the normal equation to solve directly
                         defaults to True
        '''
        self.data = np.insert(data,0,1,axis=1)
        self.targets = targets.T
        self.nsamps, self.nvars = data.shape

        if useNormEq:
            # try to invert but revert to pseudo-inverse if necessary
            try:
                xInv = np.linalg.inv(np.dot(self.data.T,self.data))

            except LinAlgError:
                xInv = np.linalg.pinv(np.dot(self.data.T,self.data))

            self.weights = np.dot(np.dot(xInv, self.data.T),targets.T)
            
        else:
            # perform gradient descent
            self.gradDescent(eta,lamb,iters)


    def gradDescent(self, eta, lamb, iters):
        '''
            This version of gradient descent is not normalized
        ''' 
        # initialize weights to small positive and negative values
        self.weights = np.random.rand(1,self.nvars+1).T * -0.1 - 0.05

        currCost = self.costF()

        for i in range(iters):
            # compute difference between hypothesis output and target values
            h = np.dot(self.data,self.weights).T - self.targets

            # compute the adjustments to each weight
            weightAdj = (eta/self.nsamps) * np.dot(h,self.data).T

            # update the weights
            self.weights = self.weights - weightAdj


    def costF(self):
        # compute the cost of the current weights
        # currently using the SSE cost measure
        return (1.0/(2*self.nsamps))*np.sum((np.dot(self.data,self.weights) - self.targets) ** 2)
        



    def predict(self, data):
        '''
        Inputs:
            - data: samples in rows
        Outputs:
            - an array of predicted values based on trained model
        '''
        if self.weights is None:
            print 'Model weights have not been trained'
        else:
            testD = np.insert(data,0,1,axis=1)
            return np.dot(testD,self.weights)
