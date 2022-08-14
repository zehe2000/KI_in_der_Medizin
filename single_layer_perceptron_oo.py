#by Zeinab Herz 884503 and Lana Bou Yehia 882821

import numpy as np

class perceptron:

    def __init__(self, l_rate, max_iter):
        """
        This is an implementation of the perceptron algorithm invented by Frank Rosenblatt in 1957.

        Parameters
        ----------
        l_rate : float
            learning rate of the perceptron. Learning rate is between 0 and 1. Larger values make the weight changes more volatile
        max_iter : int
            maximum number of iteration

        """
        self.l_rate= l_rate
        self.max_iter= max_iter
        
    def predict(self, X):
        """

        Parameters
        ----------
        X : np.array
            X is training set of s samples where
                x is n-dimensional input vector 

        Returns
        -------
        np.array
            DESCRIPTION.

        """
        # Check if predict is called during training
        if not self.train:
            ones = np.ones((len(X),1))
            X = np.hstack((X, ones)) 
        # check if dot product of X and weights are greater than 0  
        return X @ self.w > 0
    
    def fit(self, X, y):
        """
        

        Parameters
        ----------
        X : np.array
             X is training set of s samples where
                 x is n-dimensional input vector 
        y : np.array
            is the desired output value of the perceptron for that input.

        Returns
        -------
        prints epoch and loss
        

        """
        self.train = True
        # https://datascience.stackexchange.com/questions/43506/attach-1-for-the-bias-unit-in-neural-networks-what-does-it-mean
        ones = np.ones((len(X),1))
        X = np.hstack((X, ones))
        
        # weight vector already contains bias.
        self.w = np.zeros(X.shape[1])
        
        for epoch in range(self.max_iter):
            # If loss is 0 after epoch then all examples have been classified correctly.
            loss = 0
            for i, row in enumerate(X):
                prediction = self.predict(row)
                
                error = prediction - y[i] 
                loss += abs(error)
                
                # Update weights.
                self.w -= self.l_rate * error * row 
                    
            print(f"Epoch: {epoch}, Loss: {loss}")
            
            # if loss is zero so that examples are correctly classified 
            if loss==0:
                self.train = False
                return 
            
        print("Maximum number of iterations reached")
        self.train = False
        return 


if __name__ == "__main__": 
    
    model = perceptron(0.5, 10)

    X = np.array([[0, 0], [0, 1], [1, 0],[1, 1]])
    y = np.array([0,0,0,1])
    model.fit(X,y)



        
