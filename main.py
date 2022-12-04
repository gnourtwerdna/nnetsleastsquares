import numpy as np
import matplotlib.pyplot as plt
import util

def main():
    X_train, y_train = util.get_train_data()
    lambda_ = 1e-4
    weights = np.random.rand(16, 1)
    losses = util.LM(X_train, y_train, weights, lambda_)
    util.loss_plot(losses)

if __name__ == "__main__":
    main()