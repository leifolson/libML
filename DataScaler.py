'''
    Author: Clinton Olson
    Email: clint.olson2@gmail.com

    The DataScaler class provides a number of data scaling techniques
    for array-like objects where rows are samples and columns are 
    variables
'''


class DataScaler:

    # init object
    def __init__(self,data):
        self.data = data
 
    # range scaling
    # columns can be specified in the event that only certain variables
    # should be scaled or normalized
    def rangeScale(self,cols):
        # get min/max values
        minVals = self.data[:,cols].min(axis=0)
        maxVals = self.data[:,cols].max(axis=0)

        # compute range
        rangeVals = maxVals - minVals

        # scale the features
        return ((self.data[:,cols] - minVals) / float(rangeVals))

        

    # zero mean, unit variance scaling
    # use cols to specify which columns in the array to normalize
    def zeroMeanUVar(self,cols):
        return (self.data[:,cols] - self.data[:,cols].mean(axis=0)) / self.data[:,cols].var(axis=0)


    # get original data
    def getData(self):
        return self.data

    # set data
    def setData(self,data):
        self.data = pd.DataFrame(data)
