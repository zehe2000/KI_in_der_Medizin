# Example of AND operator, as described above


# Begin algorithm
def perceptron():
    alpha = 0.5
    input_data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    weights = [0, 0]
    bias = 0
    # Repeat until we minimize error
    while True:
        # Start with the weights from t-1
        new_weights =weights
        new_bias = bias

        # For each input data point
        for input_datum in input_data: 
            # Add bias (intercept) to line
            comparison = bias
            list_of_vars = input_datum[0]

            # For each variable, compute the value of the line
            for index in range(len(list_of_vars)):
                comparison += weights[index] * list_of_vars[index]

            # Obtain the correct classification and the classification of the algorithm
            correct_value = input_datum[1]
            classified_value = int(comparison > 0)

            # If the values are different, add an error to the weights and the bias
            if classified_value != correct_value:
                for index in range(len(list_of_vars)):
                    weights[index] += alpha * (correct_value - classified_value) * list_of_vars[index]
                bias += alpha * (correct_value - classified_value)
               
                

        # If there is no change in weights or bias, return
        if new_weights == weights and new_bias == bias:
            return (weights, bias)
        
perceptron()