import matplotlib.pyplot as plt
import numpy
import torch

csv = 'boston.csv'
data = numpy.genfromtxt(csv, delimiter=',')

inputs = data[:, [0, 1, 2]]
inputs = inputs.astype(numpy.float32)
inputs = torch.from_numpy(inputs)
target = data[:, 3]
target = target.astype(numpy.float32)
target = torch.from_numpy(target)


def compute_mse(y, yhat):
    yhat = torch.flatten(yhat)
    return ((yhat - y)**2) . mean()


def linear_regression(X, y, lr):
    mse_log = []
    w = torch.randn(X.shape[1], 1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    for i in range(200):
        print("Epoch", i, ":")

        # compute the model predictions
        prediction = X @ w + b
        # compute the loss and its gradient
        loss = compute_mse(y, prediction)
        loss.backward()

        print("Loss = ", loss)
        mse_log.append(loss)
        with torch.no_grad():

            # update the weight
            w -= lr * w.grad
            # update the bias
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
    return mse_log


learning_rate = [ 0.01/10**i for i in range(3)]
for lr in learning_rate:
    mse_log = linear_regression(inputs, target, lr)
    plt.plot(mse_log)
    plt.xlabel("iteration")
    plt.ylabel("Mean Squared Loss")
    title = "MSE vs Epoch with learning rate {}.png".format(lr)
    plt.title(title)
    plt.savefig(title)
    plt.show()
