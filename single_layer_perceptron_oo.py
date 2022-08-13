class perceptron:
    def __init__(self, l_rate, n_epoch ):
        self.l_rate= l_rate
        self.n_epoch= n_epoch
        
    def predict (self,weights, features_x, label_y):
        bias = weights[0]
        for i in len(features_x):
            bias += weights[i + 1] * features_x[i]* label_y[i]
            return 1.0 if bias >= 0.0 else 0.0
    
    def fit(self, features_x, label_y):
        weights = [0.0 for i in range(len (features_x ))]
        for epoch in range(self.n_epoch):
            sum_error = 0.0
            index=0
            for row in features_x:
                prediction = self.predict(weights, features_x, label_y)
                error = label_y[index] - prediction
                sum_error += error**2
                weights[0] = weights[0] + self.l_rate * error
                index+=1
                for i in range(len(features_x)):
                    weights[i + 1] = weights[i + 1] + self.l_rate * error * row[i]+ self.l_rate * error*label_y[i]
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (self.epoch, self.l_rate, sum_error))
            
            if sum_error==0:
                return weights
            
            
        return weights
    
And_Operator = perceptron(0.5, 5)
And_Operator.fit([[0, 0], [0, 1], [1, 0],[1, 1]], [0,0,0,1])

#[[0, 0, 0], [0, 1, 0], [1, 0, 0],[1, 1, 1]]

        
