import numpy as np
import matplotlib.pyplot as plt
import util
from util import *

def main():
    X_train, y_train = util.get_train_data()
    X_test, y_test = util.get_train_data(train=False)
    constant = 1e-4
    weights = np.random.rand(16, 1)
    j = jacobian(X_train, weights)
    Dh = LM_matrix(j, constant)
    check = func(X_train, weights)
    temp = sse(X_train, y_train, weights)
    print(temp)

if __name__ == "__main__":
    main()