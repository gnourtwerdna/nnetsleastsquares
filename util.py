import numpy as np

def get_train_data(train=True):
    '''
    Generates a training/testing dataset of random samples from a uniform distribution in R3.

    Parameters
    ----------
    train: Default is the training dataset. Set to false for the testing dataset.

    Returns
    -------
    tuple
        A tuple representing the dataset (X, y)
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

def func(x, w):
    f = (w[0]*tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]) 
        + w[5]*tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])
        + w[10]*tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])
        + w[15])
    return f

def grad(x, w):
    '''
    Computes the gradient of f(x).

    Parameters
    ----------
    x: 3x1 matrix
    w: 16x1 matrix

    Returns
    -------
    matrix
       16x1 matrix of partial derivatives
    '''
    dw1 = tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])
    dw2 = x[0]*w[0]*deriv_tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])
    dw3 = x[1]*w[0]*deriv_tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])
    dw4 = x[2]*w[0]*deriv_tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])
    dw5 = w[0]*deriv_tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])
    dw6 = tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])
    dw7 = x[0]*w[5]*deriv_tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])
    dw8 = x[1]*w[5]*deriv_tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])
    dw9 = x[2]*w[5]*deriv_tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])
    dw10 = w[5]*deriv_tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])
    dw11 = tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])
    dw12 = x[0]*w[10]*deriv_tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])
    dw13 = x[1]*w[10]*deriv_tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])
    dw14 = x[2]*w[10]*deriv_tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])
    dw15 = w[10]*deriv_tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])
    dw16 = 1
    return np.array([dw1, dw2, dw3, dw4, dw5, dw6, dw7, dw8, dw9, dw10, dw11, dw12, dw13, dw14, dw15, dw16],dtype=object)

def jacobian(X, w):
    '''
    Computes the Jacobian matrix.

    Parameters
    ----------
    X: 500x3 matrix

    Returns
    -------
    matrix
       500x16 matrix of gradients.
    '''
    j = []
    for x in X:
        g = grad(x, w)
        j.append(g)
    return np.array(j)

def LM_matrix(j, constant):
    '''
    Computes the LM matrix.

    Parameters
    ----------
    j: 500x16 jacobian matrix

    Returns
    -------
    matrix
       516x16 matrix of gradients and h(w).
    '''
    h = np.sqrt(constant)*np.identity(16)
    return np.concatenate((j, h))

def tanh(x):
    '''
    Computes the tanh function. f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Parameters
    ----------
    x: The internal value while a pattern goes through the network

    Returns
    -------
    float
       Value after applying tanh function
    '''
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def deriv_tanh(x):
    '''
    Computes the derivative of the tanh function. f'(x) = 1-tanh(x)^2

    Parameters
    ----------
    x: The internal value while a pattern goes through the network

    Returns
    -------
    float
       Value after applying tanh function
    '''
    return 1-np.square(tanh(x))

def RMSE(y, t):
    '''
    Compute the root mean square error. RMSE = sqrt(Σ(y-t)^2 / N)

    Parameters
    ----------
    y: The predicted values
    t: The actual values

    Returns
    -------
    float
       Value after calculating RMSE
    '''
    return np.sqrt(np.square((y - t)).mean())