# Lab 6 Softmax Classifier
import numpy as np
import matplotlib.pyplot as plt



class GradientDescent():
    def __init__(self, learning_rate=0.01, threshold=0.01, max_iterations=101):
        self._learning_rate = learning_rate
        self._threshold = threshold
        self._max_iterations = max_iterations
        self._W = None

    def fit(self, x_data, y_data):
        num_examples, num_features = np.shape(x_data)
        # print ( num_examples, num_features)
        self._W = np.ones(num_features)
        self._W = np.array([[1],[1],[1],[1]])

        # print (self._W)
        x_data_transposed = x_data.transpose()


        cost_gr =[]
        Epoch = self._max_iterations

        for i in range(self._max_iterations):
            # 실제값과 예측값의 차이
            diff = np.dot(x_data, self._W) - y_data
            # print (x_data.shape, self._W.shape, y_data.shape, diff.shape)
            # print (diff)

            # diff를 이용하여 cost 생성 : 오차의 제곱합 / 2 * 데이터 개수
            cost = np.sum(diff ** 2) / (2 * num_examples)
            # print (cost)
            cost_gr.append(cost)

            # transposed X * cost / n
            # print (x_data_transposed, x_data_transposed.shape, diff.shape)
            # print (num_examples)
            gradient = np.dot(x_data.transpose(), diff) / num_examples

            # W벡터 업데이트
            # gradient = gradient.sum(axis=1)
            self._W = self._W - self._learning_rate * gradient


            if i % 1 == 0:
                print("Step: {:5},\tCost: {:22}\t\t".format(i, cost))
                # print (self._W)

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
