import numpy as np
import matplotlib.pyplot as plt
import util
from util import *

def main():
    X_train, y_train = util.get_train_data()
    X_test, y_test = util.get_train_data(train=False)
    lambda_ = 1e-4
    weights = np.random.rand(16, 1)
    temp = LM(X_train, y_train, weights, lambda_)
    print(temp)

if __name__ == "__main__":
    main()