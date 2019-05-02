# 01 Linear regression using Gradient Descent function without Tensorflow

# import
import numpy as np
import sys
import matplotlib.pyplot as plt

# Si Photonic device
from Phase_shifters import ps
device = ps()



class GradientDescent():
    def __init__(self, learning_rate=0.01, threshold=0.01, max_iterations=101, weight=1, v_bias=0):
        self._learning_rate = learning_rate
        self._threshold = threshold
        self._max_iterations = max_iterations
        self._W = weight
        self._V = v_bias

    def fit(self, x_train, y_train):
        # Data normalization
        x_data = np.array(x_train) / max(np.concatenate((x_train, y_train)))
        y_data = np.array(y_train) / max(np.concatenate((x_train, y_train)))
        x_data = np.array(x_train)
        y_data = np.array(y_train)

        #For graph
        cost_gr = []
        weight_gr = []
        v_bias_gr = []

        if len(x_data) == len(y_data):
            no_train = len(x_data)
        else:
            print ('ERROR: number of train dose not match.')
            sys.exit()

        self._W = device.MRR(0.98, 0.98, self._V, 10)
        print (self._V, self._W)

        for i in range(self._max_iterations):
            #H:Hypothesis
            H = x_data*self._W - y_data

            #Cost function
            cost = np.sum(H ** 2) / (2 * no_train)
            cost_gr.append(cost)

            # Weight investigation
            gradient = np.sum(H * x_data) / no_train
            Weight = self._W - self._learning_rate * gradient
            # print (Weight)

            # Voltage update
            v_step = 0.0001

            while abs(Weight - device.MRR(0.98, 0.98, self._V, 10)) > 0.001:
                if Weight > device.MRR(0.98, 0.98, self._V, 10):
                    self._V = self._V - v_step
                else:
                    self._V = self._V + v_step
            v_bias_gr.append(self._V)

            # Weight update
            self._W = device.MRR(0.98, 0.98,self._V, 10)
            weight_gr.append(self._W)

            if i % 10 == 0:
                print("Step: {:5},\tCost: {:22}\t\tWeight: {:20}\t\tVoltage: {:20}".format(i, cost, self._W, self._V))

        return self._W, cost_gr, self._max_iterations, weight_gr, v_bias_gr

# X and Y data
x_train = [1, 2, 3, 4]
y_train = [3, 6, 9, 12]


GD = GradientDescent()
[W, cost_gr, Epoch, weight_gr, v_bias_gr] = GD.fit(x_train, y_train)
# plt.plot(list(range(Epoch)), cost_gr)
plt.plot(v_bias_gr, weight_gr, 'ro')

f = open("result_MRR.csv", 'w')
for i in list(range(len(v_bias_gr))):
    data = "%s\t %s\t \n" %(v_bias_gr[i], weight_gr[i])
    print (data)
    f.write(data)
f.close()



plt.show()

