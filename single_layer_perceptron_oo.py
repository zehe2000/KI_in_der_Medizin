import logging
import time
import numpy as np

class perceptron:

    def __init__(self, l_rate, max_iter):
        """
        Implementation of the perceptron algorithm invented by Frank Rosenblatt in 1957.

        Parameters
        ----------
        l_rate : float
            Learning rate of the perceptron. Learning rate is between 0 and 1.
            Larger values make the weight changes more volatile
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
            X is training set of s samples where x is n-dimensional input vector 

        Returns
        -------
        np.array
            

        """
        # Check if predict is called during training
        if not self.train:
            ones = np.ones((len(X),1))
            X = np.hstack((X, ones)) 
        # check if dot product of X and weights are greater than 0  
        return X @ self.w > 0
    
    def fit(self, X, y):
        """
        Learn weights and biases that enable the model to correctly classify samples from X.

        Parameters
        ----------
        X : np.array
             X is training set of s samples where
                 x is n-dimensional input vector 
        y : np.array
            is the desired output value of the perceptron for that input.

        Returns
        -------

        

        """
        # Keep logs of each training run.
        logging.basicConfig(filename="perceptron.log", format="%(message)s", level=logging.INFO)
        t = time.localtime()
        current_time = time.strftime("%D %H:%M", t)
        logging.info(f"Logging training run for perceptron algorithm on {current_time}")
        
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
                    
            logging.info(f"Epoch: {epoch}, Loss: {loss}")
            
            # if loss is zero so that examples are correctly classified 
            if loss==0:
                self.train = False
                logging.shutdown()
                return 
            
        logging.info(f"Maximum number of {self.max_iter} iterations reached.\n\n")
        logging.shutdown()
        self.train = False
        return 


if __name__ == "__main__": 
    
    model = perceptron(0.5, 20)

    X = np.array([[0, 0], [0, 1], [1,0], [1,1]])
    y = np.array([0,1,1,0])
    model.fit(X,y)



        
