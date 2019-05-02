# Lab 6 Softmax Classifier
import numpy as np
import matplotlib.pyplot as plt

# Si Photonic device
from Phase_shifters import ps
device = ps()
ph_device = 'MZI'

v_bias = 0
Pmax = 5
if ph_device == 'MZI':
    ph_device = device.MZI(v_bias, Pmax)
else:
    ph_device = device.MRR(0.98, 0.98, v_bias, Pmax)

class GradientDescent():
    def __init__(self, learning_rate=0.01, threshold=0.01, max_iterations=101):
        self._learning_rate = learning_rate
        self._threshold = threshold
        self._max_iterations = max_iterations
        self._W = None

    def fit(self, x_data, y_data):
        num_examples, num_features = np.shape(x_data)
        # print ( num_examples, num_features)
        # self._W = np.ones(num_features)
        # self._W = np.array([[1],[1],[1],[1]])
        self._V = np.array([[0], [0], [0], [0]])
        self._W = device.MZI(self._V, Pmax)

        # print (self._W)
        x_data_transposed = x_data.transpose()


        cost_gr =[]
        v_bias_gr =[]
        weight_gr=[]
        Epoch = self._max_iterations

        for i in range(self._max_iterations):

            diff = np.dot(x_data, self._W) - y_data
            cost = np.sum(diff ** 2) / (2 * num_examples)
            cost_gr.append(cost)

            # Weight investigation
            gradient = np.dot(x_data.transpose(), diff) / num_examples
            Weight = self._W - self._learning_rate * gradient
            # print (Weight)

            # Voltage update
            v_step = 0.0001

            print (abs(Weight - device.MZI(self._V, 10)))

            while abs(Weight - device.MZI(self._V, Pmax)) > 0.001:
                if Weight > device.MZI(self._V, Pmax):
                    self._V = self._V - v_step
                else:
                    self._V = self._V + v_step
                # print (self._V, abs(Weight - device.MZI(self._V, 10)))
            v_bias_gr.append(self._V)

            # Weight update
            self._W = device.MZI(self._V, Pmax)
            weight_gr.append(self._W)


            if i % 1 == 0:
                print("Step: {:5},\tCost: {:22}\t\t".format(i, cost))

        return self._W, cost_gr, Epoch



# Predicting animal type based on various features
xy = np.loadtxt("Test_yh2424/Iris/data-01_iris_edited.csv", delimiter=',', dtype=np.float32)
# xy = np.loadtxt("data-01_iris_edited.csv", delimiter=',', dtype=np.float32)
x_train = xy[:, 0:-1]
y_train = xy[:, [-1]]

# print(x_train.shape, y_train.shape)
#
GD = GradientDescent()
[W, cost_gr, Epoch] = GD.fit(x_train, y_train)
plt.plot(list(range(Epoch)), cost_gr, 'ro')
plt.show()



x=x_train
y=y_train

for i in list(range(150)):
    # i =
    result = x[i][0]*W[0] + x[i][1]*W[1] + x[i][2]*W[2] + x[i][3]*W[3]
    if np.round(result) == y[i]:
        answer = 1
    else:
        answer = 0
    print (i, result, np.round(result), y[i], answer)


''' 

'''
