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
new_X = []
for i in range(len(y)):
    new_X.append([1, X[i][1], X[i][2], X[i][3], X[i][1] ** 2, X[i][2] ** 2, X[i][3] ** 2, X[i][1] ** 3, X[i][2] ** 3,
                  X[i][3] ** 3])
new_X = np.array(new_X, dtype=np.float64)


def k_fold_cross_validation(xx, yy, k, lamb):
    num = 0
    train_list = []
    test_list = []
    le = int(len(yy) / k)
    for i in range(k):
        tr_x = []
        tr_y = []
        te_x = []
        te_y = []
        for k in range(0, num):
            tr_x.append(xx[k])
            tr_y.append(yy[k])
        for k in range(num, num + le):
            te_x.append(xx[k])
            te_y.append(yy[k])
        for k in range(num + le, len(yy)):
            tr_x.append(xx[k])
            tr_y.append(yy[k])
        num += le
        tr_x = np.array(tr_x, dtype=np.float64)
        tr_y = np.array(tr_y, dtype=np.float64)
        te_x = np.array(te_x, dtype=np.float64)
        te_y = np.array(te_y, dtype=np.float64)
        LRGD4 = LinearRegressionUsingGD(eta=0.01, epochs=1000)
        W2, tmp = LRGD4.fit(tr_x, tr_y, lamb)
        train_list.append(tmp)
        pred = LRGD4.predict(te_x)
        res = pred - te_y
        tmp = np.sum(res ** 2) / (2 * len(te_y))
        test_list.append(tmp)
    return np.mean(train_list), np.mean(test_list)


lambdas = []
test_errors = []
train_errors = []
for lam in np.arange(25, 1000, 25):
    train_cost, test_cost = k_fold_cross_validation(new_X, y, 10, lam)
    lambdas.append(math.log(lam, math.exp(1)))
    train_errors.append(train_cost)
    test_errors.append(test_cost)
miin = 10000
min_ind = 0
for i in range(len(train_errors)):
    if miin > train_errors[i]:
        miin = train_errors[i]
        min_ind = i

print("Best lambda is: " + str(min_ind*25+25))
fig, axs = plt.subplots(2)
axs[0].plot(lambdas, train_errors)
axs[1].plot(lambdas, test_errors)
axs[0].legend(['Train errors vs ln(lambda)'])
axs[1].legend(['Test errors vs ln(lambda)'])
plt.show()
