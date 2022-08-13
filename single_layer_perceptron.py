def predict(row, weights):
	bias = weights[0]
	for i in range(len(row)-1):
		bias += weights[i + 1] * row[i]
	return 1.0 if bias >= 0.0 else 0.0
 
# Estimate Perceptron weights using stochastic gradient descent
#fit
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights
 
# Calculate weights
dataset = [[0, 0, 0], [0, 1, 0], [1, 0, 0],[1, 1, 1]]
# dataset = [[features, label]...]
l_rate = 0.5
n_epoch = 10
weights = train_weights(dataset, l_rate, n_epoch)
print(weights)
predict(dataset[0], weights)
