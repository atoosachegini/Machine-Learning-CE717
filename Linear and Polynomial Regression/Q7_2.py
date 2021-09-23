import numpy as np
import math
import matplotlib.pyplot as plt


class LinearRegressionUsingGD:
    def __init__(self, eta=0.01, epochs=5000):
        self.cost_ = []
        self.eta = eta
        self.epochs = epochs

    def fit(self, x, y, lambda1):
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]
        for _ in range(self.epochs):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            if lambda1 == -1:
                gradient_vector1 = (1 / m) * np.dot(x.T, residuals)
            else:
                gradient_vector1 = (1 / m) * (np.dot(x.T, residuals) + lambda1 * self.w_)
            self.w_ -= (self.eta / m) * np.nan_to_num(gradient_vector1)
            cost = np.sum(residuals ** 2) / (2 * m)
            if lambda1 != -1:
                cost += (lambda1 / (2 * m)) * np.sum(np.square(self.w_))
            self.cost_.append(cost)
        return self.w_, cost

    def predict(self, x):
        return np.dot(x, self.w_)


X = []
y = []
X1 = []
X2 = []
X3 = []
with open("SecondDataset.txt", "r") as a_file:
    for line in a_file:
        stripped_line = line.strip().split()
        X1.append(float(stripped_line[0]))
        X2.append(float(stripped_line[1]))
        X3.append(float(stripped_line[2]))
        y.append(float(stripped_line[3]))
minn = min(X1)
maxx = max(X1)
for i in range(len(X1)):
    X1[i] = (X1[i] - minn) / (maxx - minn)
minn = min(X2)
maxx = max(X2)
for i in range(len(X2)):
    X2[i] = (X2[i] - minn) / (maxx - minn)
minn = min(X3)
maxx = max(X3)
for i in range(len(X3)):
    X3[i] = (X3[i] - minn) / (maxx - minn)
minn = min(y)
maxx = max(y)
for i in range(len(y)):
    y[i] = [(y[i] - minn) / (maxx - minn)]
for j in range(len(y)):
    X.append([1, X1[j], X2[j], X3[j]])
X = np.array(X, dtype=np.float64)
y = np.array(y, dtype=np.float64)
### first one
train_x = []
train_y = []
test_x = []
test_y = []
length = len(y)
for i in range(int(0.2 * length)):
    test_x.append(X[i])
    test_y.append(y[i])
for j in range(int(0.2 * length) + 1, length):
    train_x.append(X[j])
    train_y.append(y[j])
train_x = np.array(train_x, dtype=np.float64)
train_y = np.array(train_y, dtype=np.float64)
test_x = np.array(test_x, dtype=np.float64)
test_y = np.array(test_y, dtype=np.float64)
LRGD = LinearRegressionUsingGD(eta=0.01, epochs=1000)
W1, train_cost = LRGD.fit(train_x, train_y, lambda1=-1)
print("Predicted W: " + str(W1))
print("Train error is: " + str(train_cost))
predicted_y = LRGD.predict(test_x)
residuals1 = predicted_y - test_y
test_cost = np.sum(residuals1 ** 2) / (2 * len(test_y))
print("Test error is: " + str(test_cost))

### second one

new_X = []
for i in range(len(y)):
    new_X.append([1, X[i][1], X[i][2], X[i][3], X[i][1] ** 2, X[i][2] ** 2, X[i][3] ** 2, X[i][1] ** 3, X[i][2] ** 3,
                  X[i][3] ** 3])
new_X = np.array(new_X, dtype=np.float64)
train_x = []
train_y = []
test_x = []
test_y = []
for i in range(int(0.2 * length)):
    test_x.append(new_X[i])
    test_y.append(y[i])
for j in range(int(0.2 * length) + 1, length):
    train_x.append(new_X[j])
    train_y.append(y[j])
train_x = np.array(train_x, dtype=np.float64)
train_y = np.array(train_y, dtype=np.float64)
test_x = np.array(test_x, dtype=np.float64)
test_y = np.array(test_y, dtype=np.float64)
LRGD2 = LinearRegressionUsingGD(eta=0.01, epochs=1000)
W1, train_cost = LRGD2.fit(train_x, train_y, lambda1=-1)
print("Predicted W: " + str(W1))
print("Train error is: " + str(train_cost))
predicted_y = LRGD2.predict(test_x)
residuals1 = predicted_y - test_y
test_cost = np.sum(residuals1 ** 2) / (2 * len(test_y))
print("Test error is: " + str(test_cost))

### third one
LRGD3 = LinearRegressionUsingGD(eta=0.01, epochs=1000)
W1, train_cost = LRGD3.fit(train_x, train_y, 100)
print("Predicted W: " + str(W1))
print("Train error is: " + str(train_cost))
predicted_y = LRGD3.predict(test_x)
residuals1 = predicted_y - test_y
test_cost = np.sum(residuals1 ** 2) / (2 * len(test_y))
print("Test error is: " + str(test_cost))
lambdas = []
test_errors = []
train_errors = []
for lam in np.arange(25, 1000, 25):
    LRGD3 = LinearRegressionUsingGD(eta=0.01, epochs=1000)
    W1, train_cost = LRGD3.fit(train_x, train_y, lam)
    predicted_y = LRGD3.predict(test_x)
    residuals1 = predicted_y - test_y
    test_cost = np.sum(residuals1 ** 2) / (2 * len(test_y))
    lambdas.append(math.log(lam, math.exp(1)))
    train_errors.append(train_cost)
    test_errors.append(test_cost)


fig, axs = plt.subplots(2)
axs[0].plot(lambdas, train_errors)
axs[1].plot(lambdas, test_errors)
axs[0].legend(['Train errors vs ln(lambda)'])
axs[1].legend(['Test errors vs ln(lambda)'])
plt.show()