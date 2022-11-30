import numpy as np

def data_generator(train=True):
    '''
    Generates a training/testing dataset of random samples from a uniform distribution in R3.

    Parameters
    ----------
    train: Default is the training dataset. Set to false for the testing dataset.

    Returns
    -------
    tuple
        A tuple representing the dataset (X, y).
    '''
    if train == True:
        X = np.random.rand(500, 3)
        y = np.zeros((500, 1))
        for i in range(len(X)):
            y[i] = X[i][0] * X[i][1] + X[i][2]
    if train == False:
        X = np.random.rand(100, 3)
        y = np.zeros([100, 1])
        for i in range(len(X)):
            y[i] = X[i][0] * X[i][1] + X[i][2]
    return X, y