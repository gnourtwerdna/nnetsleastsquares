import numpy as np
import matplotlib.pyplot as plt
import util
from network import jacobian, LM_matrix
import network

def main():
    X_train, y_train = util.get_train_data()
    X_test, y_test = util.get_train_data(train=False)
    constant = 1e-4
    weights = np.random.rand(16, 1)
    j = jacobian(X_train, weights)
    Dh = LM_matrix(j, constant)
    print(Dh.shape)
if __name__ == "__main__":
    main()