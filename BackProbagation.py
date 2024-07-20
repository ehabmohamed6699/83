import random
import numpy as np


labels = ["BOMBAI", "CALI", "SIRA"]

class Neuron:
    def __init__(self, in_weights, out_weights, activate = None, inv_activate = None):
        self.out_weights = out_weights
        self.in_weights = in_weights
        # self.weights = weights
        self.output = 0
        self.local_error = 0
        self.activate = activate
        self.inv_activate = inv_activate

    def calc(self, first, input):
        self.input = input
        if first:
            self.output = self.input
            return self.output
        try:
            self.output = self.activate(np.dot(self.in_weights, input))
        except:
            print("error occured")
        return self.output

    def error(self, last, desired, errors):
        if last:
            self.local_error = (desired - self.output) * self.inv_activate(self.output)
        else:
            self.local_error = self.inv_activate(self.output)*np.dot(self.out_weights, errors)
        return self.local_error
    def update_weights(self, learning_rate):
        for i in range(len(self.in_weights)):
            self.in_weights[i] += (learning_rate * self.local_error * self.input[i])

class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

    def output(self):
        self.output_vector = [x.output for x in self.neurons]
        return self.output_vector

class MLPClassifier:
    def __init__(self, input_size, hidden_layers, output_size, activate, inv_activate, bias):
        self.bias = bias
        self.confusion_matrix = np.zeros((output_size, output_size))
        hidden_layers.append(output_size)
        model = []
        curr_layer = input_size
        out_weights = np.random.rand(hidden_layers[0])* (1+1) -1
        neurons = []
        for i in range(curr_layer):
            neurons.append(Neuron(None, np.random.rand(hidden_layers[0])* (1+1) -1, activate, inv_activate))
        model.append(Layer(neurons))
        curr_layer = hidden_layers[0]
        for i in range(1, len(hidden_layers)):
            #in_weights = out_weights
            out_weights = np.random.rand(hidden_layers[i])* (1+1) -1
            neurons = []
            for j in range(curr_layer):
                if not bias:
                    neurons.append(Neuron([x.out_weights[j] for x in model[len(model) - 1].neurons],
                                          np.random.rand(hidden_layers[i])* (1+1) -1, activate, inv_activate))
                else:
                    temp = [random.random()]
                    temp.extend([x.out_weights[j] for x in model[len(model) - 1].neurons])
                    neurons.append(Neuron(temp,
                                          np.random.rand(hidden_layers[i])* (1+1) -1, activate, inv_activate))
            model.append(Layer(neurons))
            curr_layer = hidden_layers[i]
        in_weights = out_weights
        neurons = []
        for i in range(curr_layer):
            if bias:
                temp = [random.random()]
                temp.extend([x.out_weights[i] for x in model[len(model) - 1].neurons])
                neurons.append(
                    Neuron(temp, None,
                           activate, inv_activate))
            else:
                neurons.append(
                    Neuron([x.out_weights[i] for x in model[len(model) - 1].neurons], None,
                           activate, inv_activate))
        model.append(Layer(neurons))
        self.architecture = model
    def learn(self, inputs, epochs, threshold, rate):
        epoch = 0

        while epoch < epochs:
            mse = 0
            epoch += 1
            for input in inputs:
                curr_input = input[0]
                if self.bias:
                    temp = curr_input
                    curr_input = [1]
                    curr_input.extend(temp)
                for i in range(len(self.architecture[0].neurons)):
                    self.architecture[0].neurons[i].calc(True, curr_input[i])
                for i in range(1, len(self.architecture)):
                    for j in range(len(self.architecture[i].neurons)):
                        self.architecture[i].neurons[j].calc(False, curr_input)
                    if self.bias:
                        temp = [x.output for x in self.architecture[i].neurons]
                        curr_input = [1]
                        curr_input.extend(temp)
                        # curr_input = [1].extend()
                    else:
                        curr_input = [x.output for x in self.architecture[i].neurons]
                for i in range(len(self.architecture) - 1, 0, -1):
                    last = i == len(self.architecture) - 1
                    for j in range(len(self.architecture[i].neurons)):
                        if last:
                            mse += self.architecture[i].neurons[j].error(last, input[1][j], 0) * self.architecture[i].neurons[j].error(last, input[1][j], 0)
                        else:
                            self.architecture[i].neurons[j].error(last, 0, [x.local_error for x in self.architecture[i + 1].neurons])
                for i in range(1, len(self.architecture)):
                    for j in range(len(self.architecture[i].neurons)):
                        self.architecture[i].neurons[j].update_weights(rate)
            if mse <= threshold:
                break

    def test(self, input):
        curr_input = input[0]
        if self.bias:
            temp = curr_input
            curr_input = [1]
            curr_input.extend(temp)
        for i in range(len(self.architecture[0].neurons)):
            self.architecture[0].neurons[i].calc(True, curr_input[i])
        for i in range(1, len(self.architecture)):
            for j in range(len(self.architecture[i].neurons)):
                self.architecture[i].neurons[j].calc(False, curr_input)
            if self.bias:
                temp = [x.output for x in self.architecture[i].neurons]
                curr_input = [1]
                curr_input.extend(temp)
            else:
                curr_input = [x.output for x in self.architecture[i].neurons]
        return self.architecture[len(self.architecture) - 1].output()

    def predict(self, input, show = False, update = False):
        result = self.test(input)
        max = 0
        for i in range(1, len(result)):
            if result[i] > result[max]:
                max = i
        if show:
            print("Desired: ", labels[input[1].index(1)])
            print("Actual: ", labels[max])
        if update:
            self.confusion_matrix[input[1].index(1), max] += 1

        return input[1][max] == 1


    def evaluate(self, test):
        mse = 0
        misclass = 0

        for input in test:
            predicted = self.test(input)
            if not self.predict(input, update=True):
                misclass += 1
            for i in range(len(predicted)):
                mse += (input[1][i] - predicted[i]) * (input[1][i] - predicted[i])
        mse /= len(test)
        print("MEAN ERROR: ", mse)
        print("NO. OF MISSCLASSIFICATIONS: ", misclass)
        print("CLASSIFICATION ERROR: ", misclass / len(test))
        print("CLASSIFICATION ACCURACY: ", ((len(test) - misclass) / len(test)) * 100)
        print("CONFUSION MATRIX: ")
        for item in self.confusion_matrix:
            print(item[0], "|", item[1], "|", item[2])
            print("---------------------")
        self.confusion_matrix = np.zeros((3,3))
