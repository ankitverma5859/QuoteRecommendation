import pandas as pd


class Perceptron:

    def __init__(self, training_data, test_data):
        l_rate = 0.01
        n_epoch = 500
        #scores = self.evaluate_algorithm(training_data,test_data, self.perceptron, l_rate, n_epoch)
        #print('Scores: %s' % scores)
        #print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, training_data, test_data, algorithm, *args):
        scores = list()

        test_set = list()
        for row in test_data:
            row_copy = list(row)
            row_copy.pop()
            test_set.append(row_copy)

        predicted = algorithm(training_data, test_set, *args)
        actual = [row[-1] for row in test_data]
        accuracy = self.accuracy_metric(actual, predicted)
        scores.append(accuracy)

        return scores

    # Make a prediction with weights
    def predict(self, row, weights):
        activation = weights[0]
        for i in range(len(row) - 1):
            activation += weights[i + 1] * row[i]
        return 1.0 if activation >= 0.0 else 0.0

    # Estimate Perceptron weights using stochastic gradient descent
    def train_weights(self, train, l_rate, n_epoch):
        weights = [0.0 for i in range(len(train[0]))]
        for epoch in range(n_epoch):
            for row in train:
                prediction = self.predict(row, weights)
                error = row[-1] - prediction
                weights[0] = weights[0] + l_rate * error
                for i in range(len(row) - 1):
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        return weights

    # Perceptron Algorithm With Stochastic Gradient Descent
    def perceptron(self, train, test, l_rate, n_epoch):
        predictions = list()
        weights = self.train_weights(train, l_rate, n_epoch)
        print(weights)
        for row in test:
            prediction = self.predict(row, weights)
            predictions.append(prediction)
        return predictions

    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1

        return correct / float(len(actual)) * 100.0
