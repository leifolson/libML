'''
Author: Clinton Olson
Contact: clint.olson2@gmail.com

LogReg implements simple logistic regression

Currently only does two-class logistic regression
'''
import numpy as np

class LogReg:

    def __init__(self):
        self.data = None
        self.targets = None
        self.nsamps = 0
        self.nvars = 0
        self.weights = None
        

    def train(self, data, targets, eta = 0.2, lamb = 0, iters = 100):
        '''
        Inputs:
            - data: samples in rows, vars in columns
            - targets: array of target values
            - eta: learning rate, default = 0.2
            - lamb: regularzation term, default = 0  i.e., no regularization
        '''
        self.data = np.insert(data,0,1,axis=1)
        self.targets = targets.T
        self.nsamps, self.nvars = data.shape

        # perform gradient descent
        self.gradDescent(eta,lamb,iters)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def gradDescent(self, eta, lamb, iters):
        '''
            This version of gradient descent is not regularized
        ''' 
        # initialize weights to small positive and negative values
        self.weights = np.random.rand(1,self.nvars+1).T * 0.1 - 0.05

        currCost = self.costF()

        for i in range(iters):
            # compute difference between hypothesis output and target values
            h = self.sigmoid(np.dot(self.data,self.weights).T) - self.targets

            # compute the adjustments to each weight
            weightAdj = (eta/self.nsamps) * np.dot(h,self.data).T

            # update the weights
            self.weights = self.weights - weightAdj

            print self.costF()


    def costF(self):
        # compute the cost of the current weights
        h = self.sigmoid(np.dot(self.data,self.weights))
        cost = -self.targets*np.log(h.T) - (1-self.targets)*np.log(1-h.T)
        return np.sum(cost)
        
        



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
            return self.sigmoid(np.dot(testD,self.weights))
