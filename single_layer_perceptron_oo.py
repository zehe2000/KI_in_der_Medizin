import numpy as np

class perceptron:
    """ This is an implementation of the perceptron algorithm """
    def __init__(self, l_rate, max_iter):
        self.l_rate= l_rate
        self.max_iter= max_iter
        
    def predict(self, X):
        # Check if predict is called during training
        if not self.train:
            ones = np.ones((len(X),1))
            X = np.hstack((X, ones)) 
            
        return X @ self.w > 0
    
    def fit(self, X, y):
        self.train = True
        # Append one-vector for bias-trick.
        ones = np.ones((len(X),1))
        X = np.hstack((X, ones))
        # weight vector already contains bias.
        self.w = np.zeros(X.shape[1])
        
        for epoch in range(self.max_iter):
            sum_error = 0.0
            index=0
            for row in X:
                prediction = self.predict(row)
                error = y[index] - prediction
                sum_error += error**2
                self.w[0] = self.w[0] + self.l_rate * error
                index+=1
                for i in range(X.shape[1]):
                    self.w[i] = self.w[i] + self.l_rate * error * row[i]+ self.l_rate * error*y[i]
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.l_rate, sum_error))
            
            if sum_error==0:
                self.train = False
                return 
        
        self.train = False
        return 

if __name__ == "__main__": 
    
    model = perceptron(0.5, 10)

    X = np.array([[0, 0], [0, 1], [1, 0],[1, 1]])
    y = np.array([0,0,0,1])
    model.fit(X,y)



        
