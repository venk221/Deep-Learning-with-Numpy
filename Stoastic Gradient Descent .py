import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X_train = np.load(r"D:\WPI\Academics\Deep_Learning\Deep-Learning-Projects\Project_3\fashion_mnist_train_images.npy")
y_train = np.load(r"D:\WPI\Academics\Deep_Learning\Deep-Learning-Projects\Project_3\fashion_mnist_train_labels.npy").reshape(-1,1)
X_test = np.load(r"D:\WPI\Academics\Deep_Learning\Deep-Learning-Projects\Project_3\fashion_mnist_test_images.npy")
y_test = np.load(r"D:\WPI\Academics\Deep_Learning\Deep-Learning-Projects\Project_3\fashion_mnist_test_labels.npy").reshape(-1,1)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
X_train = np.transpose(X_train)
X_valid = np.transpose(X_valid)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

def one_hot_encoder(Y):
    ohe_train = np.zeros((Y.shape[0], 10))
    for k, i in enumerate(range(ohe_train.shape[0])):
        ohe_train[i, Y[k]] = 1

    return ohe_train

y_train_encoded = one_hot_encoder(y_train)
print(y_train_encoded.shape)
y_valid_encoded = one_hot_encoder(y_valid)
y_test_encoded = one_hot_encoder(y_test)
print(y_valid_encoded.shape)

## Random initializations for hyper-parameters
w = np.random.randn(X_train.shape[0], y_train_encoded.shape[1]) * 0.01
bias = np.random.random(10)
num_epoch = [1, 2]  # [3, 4]
epsilon = [0.000003]  # [0.000004,0.000005,0.000006]
alpha = [2]  # [3, 4, 5]
mini_batch_size = [200, 100]  # [600, 400]


print(f'X_train = {X_train.shape}\ny_train_encoded = {y_train_encoded.shape}\nweights = {w.shape}')


def softmax1(z):
    exp = np.exp(z - np.max(z, axis=0))
    return exp / np.sum(exp, axis=0)


def grad_w(X, w, Y, b, alpha):
    y = X.T @ w + b
    y_pred = softmax1(y)
    
    y_hat = y_pred - Y
    reg = (alpha * w) / X.shape[1]
    error = np.dot(X, y_hat) + reg
    return error


def grad_b(X, w, Y, b):
    y = X.T @ w + b
    y_pred = softmax1(y)
    y_hat = y_pred - Y
    error = y_hat / X.shape[1]
    return error


def Fce(X, w, Y, b, alpha):
    z = np.dot(X.T, w) + b
    exp_Z = np.exp(z)
    exp_Z_mean = np.reshape(np.sum(exp_Z, axis=1), (-1, 1))
    Y_hat = exp_Z / (exp_Z_mean + 1e-10)
    logY_hat = np.log(Y_hat)
    loss = -np.sum(Y * logY_hat) / X.shape[1]
    Fce = loss
    return Fce


def SGD(epoch, lr, alpha, mb, X_train, w, b):
    for k, ep in enumerate(range(epoch)):
        if k < 2:
            mini_batch_size = X_train.shape[1] // mb
            start = 0
            end = int(mini_batch_size)
            for i in range(mb):
                X = X_train[:, start:end]
                Y = y_train_encoded[start:end, :]
                dw = grad_w(X, w, Y, b, alpha)
                db = grad_b(X, w, Y, b)
                new_w = w - lr * dw
                new_b = b - lr * db
                start = end
                end = end + int(mini_batch_size)
                w = new_w
                b = new_b
            Fce_per_epoch = Fce(X, w, Y, b, alpha)
            reg_term = (alpha / 2) * (np.sum(np.dot(w.T, w)))
            Fce_per_epoch += reg_term
            print(f"Fce per epoch is={Fce_per_epoch}")
    return Fce_per_epoch


def trainer(epochs, lr, alphas, mini_batch, X_train, w, b):
    for e in epochs:
        for lrs in lr:
            for alpha in alphas:
                for mb in mini_batch:
                    SGD(e, lrs, alpha, mb, X_train, w, b)


trainer(num_epoch, epsilon, alpha, mini_batch_size, X_train, w, bias)
