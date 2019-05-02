# 01 Linear regression using Gradient Descent function without Tensorflow

# import
import numpy as np
import sys
import matplotlib.pyplot as plt

# # Si Photonic device
# from Phase_shifters import ps
# device = ps

class GradientDescent():
    def __init__(self, learning_rate=0.01, threshold=0.01, max_iterations=101, weight=10):
        self._learning_rate = learning_rate
        self._threshold = threshold
        self._max_iterations = max_iterations
        self._W = weight

    def fit(self, x_train, y_train):

        #Data normalization
        x_data = np.array(x_train)/max(np.concatenate((x_train, y_train)))
        y_data = np.array(y_train)/max(np.concatenate((x_train, y_train)))
        x_data = np.array(x_train)
        y_data = np.array(y_train)

        print (x_data, y_data)

        #For graph
        cost_gr = []


        if len(x_data) == len(y_data):
            no_train = len(x_data)
        else:
            print ('ERROR: number of train dose not match.')
            sys.exit()


        for i in range(self._max_iterations):

            #H:Hypothesis
            H = x_data*self._W - y_data

            #Cost function
            cost = np.sum(H ** 2) / (2 * no_train)
            cost_gr.append(cost)

            # Weight update
            gradient = np.sum(H * x_data) / no_train
            self._W = self._W - self._learning_rate * gradient

            if i % 10 == 0:
                print("Step: {:5},\tCost: {:22}\t\tWeight: {:20}".format(i, cost, self._W))


        return self._W, cost_gr, self._max_iterations

# X and Y data
x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]


GD = GradientDescent()
[W, cost_gr, Epoch] = GD.fit(x_train, y_train)
plt.plot(list(range(Epoch)), cost_gr)

plt.show()