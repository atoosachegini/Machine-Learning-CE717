import numpy as np


class LinearRegressionUsingGD:
    def __init__(self, eta=0.05, epochs=1000):
        self.cost_ = []
        self.eta = eta
        self.epochs = epochs

    def fit(self, x, y):
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.epochs):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)
        return self.w_

    def predict(self, x):
        return np.dot(x, self.w_)


X = []
y = []
with open("FirstDataset.txt", "r") as a_file:
    for line in a_file:
        stripped_line = line.strip().split()
        X.append([1, float(stripped_line[0])])
        y.append([float(stripped_line[1])])
X = np.array(X)
y = np.array(y)

LRGD = LinearRegressionUsingGD()
W = LRGD.fit(X, y)
print("Linear regression using gradient descent")
print("W0 is " + str(W[0][0]) + " and w1 is " + str(W[1][0]))


################
def get_best_param(X, y):
    X_transpose = X.T
    best_params = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

    return best_params  # returns a list


W = get_best_param(X, y)
print("Linear regression using closed form")
print("W0 is " + str(W[0][0]) + " and w1 is " + str(W[1][0]))
# تفاوت های این دو روش:
# 1) عملیات ماتریسی که در closed form انجام میشود در dataset های بزرگ بسیار هزینه زیادی دارد در حالی که
# با استفاده از gd محاسبات بسیار سریع تر است.
# با استفاده از gd محاسبات خیلی منعطف تر هستند مثلا میتوان با تنظیم epoch و learning rate مشخص کرد که چقدر
# زمان صرف شود و چقدر با دقت پیشروی کنیم.
