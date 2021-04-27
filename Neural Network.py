import numpy as np

class Perceptron:

    def __init__(self, inputs, bias = 1):
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias

    def set_weights(self, w_init):
        self.weights = np.array(w_init)

    def run(self, input):
        input = np.append(input, self.bias)
        return self.sigmoid(np.dot(input, self.weights))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class NeuralNetwork:

    def __init__(self, layers, bias=1, rate = 1):
        self.layers = np.array(layers, dtype=object)
        self.network = []
        self.values = []
        self.errorTerms = []

        self.bias = bias
        self.rate = rate

        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.errorTerms.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.errorTerms[i] = [0.0 for j in range(self.layers[i])]
            if i > 0:
                for j in range(self.layers[i]):
                    self.network[i].append(Perceptron(self.layers[i - 1], self.bias))

        self.network = np.array([np.array(x) for x in self.network], dtype= object)
        self.values = np.array([np.array(x) for x in self.values], dtype= object)
        self.errorTerms = np.array([np.array(x) for x in self.errorTerms], dtype= object)

    def set_weights(self, w_init):
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i + 1][j].set_weights(w_init[i][j])

    def run(self, input):
        input = np.array(input, dtype= object)
        self.values[0] = input
        for i in range(1, len(self.network)):
            for j in range(len(self.network[i])):
                self.values[i][j] = self.network[i][j].run(self.values[i - 1])

        return self.values[-1]

    def backpropagation(self, input, expected):
        input = np.array(input)
        expected = np.array(input)

        output = self.run(input)

        error = (expected - output)

        MSE = sum(error**2) / len(output)

        self.errorTerms[-1] = output * (1 - output) * error

        for i in reversed(range(1, len(self.network) - 1)):
            for j in range(self.layers[i]):
                fwd_error = 0.0
                for k in range(self.layers[i + 1]):
                    fwd_error += self.network[i + 1][k].weights[j] * self.errorTerms[i + 1][k]
                self.errorTerms[i][j] = self.values[i][j] * (1 - self.values[i][j]) * fwd_error

        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                for k in range(self.layers[i - 1] + 1):
                    if k == self.layers[i - 1]:
                        change = self.rate * self.errorTerms[i][j] * self.bias
                    else:
                        change = self.rate * self.errorTerms[i][j] * self.values[i - 1][k]
                    self.network[i][j].weights[k] += change

        return MSE       


nn = NeuralNetwork(layers=[2,4,4,1])
for i in range(1000):
    nn.backpropagation([0,0],[0])
    nn.backpropagation([0,1],[0])
    nn.backpropagation([1,0],[1])
    nn.backpropagation([1,1],[1])
    
print("NN:")
print ("0 0 = {0:.10f}".format(nn.run([0,0])[0]))
print ("0 1 = {0:.10f}".format(nn.run([0,1])[0]))
print ("1 0 = {0:.10f}".format(nn.run([1,0])[0]))
print ("1 1 = {0:.10f}".format(nn.run([1,1])[0]))