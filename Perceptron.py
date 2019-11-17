import math
import random

import pandas as pd


# Class that is the perceptron, initialized to values in the Perl slides with modifications made based on documentation
class Perceptron:
    # initializes
    cycles = 10000
    input_num = 2
    alpha = 0.04
    totalError = 0
    activation = 'Hard'

    # requires a definition of activation before running
    def __init__(self, activation, input_num, weights=None):
        if weights is None:
            weights = [random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(
                -.5, .5)]
        self.activation = activation
        self.input_num = input_num
        self.weights = weights

    # activation function, modified for unipolar
    def signal(self, net, activation='Hard'):
        if self.activation == 'Soft':
            k = 0.5  # gain
            x = net
            r = 1 / float(1 + math.exp(-(k * x)))
            return r
        elif self.activation == 'Hard':
            x = net
            if x > 0:
                y = 1
            else:
                y = 0
            return y
        elif self.activation == 'Linear':
            return net
        else:
            raise ValueError('Incorrect Input')

    def return_points(self, training_data):
        df = pd.DataFrame(columns=['Hour', 'Prediction'])
        for i in range(len(training_data)):
            if self.input_num == 0:
                df = df.append({'Hour': (training_data['Hour'][i]),
                                'Prediction': self.weights[0] * training_data['Hour'][i] + self.weights[3]},
                               ignore_index=True)
            elif self.input_num == 1:
                df = df.append({'Hour': (training_data['Hour'][i]),
                                'Prediction': self.weights[1] * training_data['Hour'][i] ** 2 + self.weights[0] *
                                              training_data['Hour'][i] + self.weights[3]}, ignore_index=True)
            elif self.input_num == 2:
                df = df.append({'Hour': (training_data['Hour'][i]),
                                'Prediction': self.weights[2] * training_data['Hour'][i] ** 3 + self.weights[1] *
                                              training_data['Hour'][i] ** 2 + self.weights[0] * training_data['Hour'][
                                                  i] + self.weights[3]}, ignore_index=True)
        return df

    # training function as implemented in slides, with slight modification to define pattern as input
    def train(self, training_data):
        for i in range(self.cycles):
            self.TotalError = 0
            for j in range(len(training_data)):
                net = 0
                if self.input_num == 0:
                    net = net + self.weights[0] * training_data['Hour'][j]
                elif self.input_num == 1:
                    net = net + self.weights[0] * training_data['Hour'][j]
                    net = net + self.weights[1] * (training_data['Hour'][j] ** 2)
                elif self.input_num == 2:
                    net = net + self.weights[0] * training_data['Hour'][j]
                    net = net + self.weights[1] * (training_data['Hour'][j] ** 2)
                    net = net + self.weights[2] * (training_data['Hour'][j] ** 3)
                else:
                    raise ValueError('Incorrect Input')
                net = self.weights[3] + net
                out = self.signal(net)
                error = training_data['Energy Consumption'][j] - out
                self.TotalError = self.totalError + (error ** 2)
                learn = self.alpha * error
                self.print_data(i, j, net, error, learn, self.totalError)
                if self.input_num == 0:
                    self.weights[0] = self.weights[0] + 2 * learn * training_data['Hour'][j]
                elif self.input_num == 1:
                    self.weights[0] = self.weights[0] + 2 * learn * training_data['Hour'][j]
                    self.weights[1] = self.weights[1] + 2 * learn * training_data['Hour'][j] ** 2
                elif self.input_num == 2:
                    self.weights[0] = self.weights[0] + 2 * learn * training_data['Hour'][j]
                    self.weights[1] = self.weights[1] + 2 * learn * training_data['Hour'][j] ** 2
                    self.weights[2] = self.weights[2] + 2 * learn * training_data['Hour'][j] ** 3
                self.weights[3] = self.weights[3] + learn * training_data['Hour'][j]
            if self.TotalError <= 0.001:
                break
        return self.return_points(training_data)

    # easy-access print data function
    def print_data(self, ite, p, net, err, lrn, tot):
        print(
            "ite={0:3} p={1} net={2:5.2f} err ={3:6.3f} lrn ={4:6.3f} tot ={5:6.3f} wei: {6} ".format(ite, p, net, err,
                                                                                                      lrn, tot,
                                                                                                      str(
                                                                                                          self.weights)))
