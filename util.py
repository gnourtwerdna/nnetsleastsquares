import numpy as np
import matplotlib.pyplot as plt

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

def func(X, w, activation_function):
    '''
    Applies f(x) on the dataset.

    Parameters
    ----------
    X: 500x3 matrix
    w: 16x1 matrix

    Returns
    -------
    np.array
        500x1
    '''
    funcs = []
    if activation_function == 'tanh':
        for x in X:
            f = (w[0]*tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]) 
                + w[5]*tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])
                + w[10]*tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])
                + w[15])
            funcs.append(f)
    if activation_function == 'sigmoid':
        for x in X:
            f = (w[0]*sigmoid(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]) 
                + w[5]*sigmoid(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])
                + w[10]*sigmoid(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])
                + w[15])
            funcs.append(f)
    return np.array(funcs)

def calculate_loss(X, y, w, lambda_, activation_function):
    '''
    Calculates l(w)=Σ(r)^2 + lambda||w||_2^2

    Parameters
    ----------
    X: 500x3 matrix
    y: 500x1 matrix
    w: 16x1 matrix

    Returns
    -------
    float
    '''
    temp = func(X, w, activation_function) - y
    temp2 = np.sqrt(lambda_) * w
    h = np.concatenate((temp, temp2))
    return (h**2).sum()
    
        
def grad(x, w, activation_function):
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
    if activation_function == 'tanh':
        dw1 = tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])[0]
        dw2 = (x[0]*w[0]*deriv_tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]))[0]
        dw3 = (x[1]*w[0]*deriv_tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]))[0]
        dw4 = (x[2]*w[0]*deriv_tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]))[0]
        dw5 = (w[0]*deriv_tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]))[0]
        dw6 = tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])[0]
        dw7 = (x[0]*w[5]*deriv_tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9]))[0]
        dw8 = (x[1]*w[5]*deriv_tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9]))[0]
        dw9 = (x[2]*w[5]*deriv_tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9]))[0]
        dw10 = (w[5]*deriv_tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9]))[0]
        dw11 = tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])[0]
        dw12 = (x[0]*w[10]*deriv_tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14]))[0]
        dw13 = (x[1]*w[10]*deriv_tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14]))[0]
        dw14 = (x[2]*w[10]*deriv_tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14]))[0]
        dw15 = (w[10]*deriv_tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14]))[0]
        dw16 = 1
    if activation_function == 'sigmoid':
        dw1 = sigmoid(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])[0]
        dw2 = (x[0]*w[0]*deriv_sigmoid(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]))[0]
        dw3 = (x[1]*w[0]*deriv_sigmoid(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]))[0]
        dw4 = (x[2]*w[0]*deriv_sigmoid(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]))[0]
        dw5 = (w[0]*deriv_sigmoid(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]))[0]
        dw6 = sigmoid(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])[0]
        dw7 = (x[0]*w[5]*deriv_sigmoid(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9]))[0]
        dw8 = (x[1]*w[5]*deriv_sigmoid(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9]))[0]
        dw9 = (x[2]*w[5]*deriv_sigmoid(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9]))[0]
        dw10 = (w[5]*deriv_sigmoid(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9]))[0]
        dw11 = sigmoid(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])[0]
        dw12 = (x[0]*w[10]*deriv_sigmoid(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14]))[0]
        dw13 = (x[1]*w[10]*deriv_sigmoid(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14]))[0]
        dw14 = (x[2]*w[10]*deriv_sigmoid(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14]))[0]
        dw15 = (w[10]*deriv_sigmoid(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14]))[0]
        dw16 = 1
    return np.array([dw1, dw2, dw3, dw4, dw5, dw6, dw7, dw8, dw9, dw10, dw11, dw12, dw13, dw14, dw15, dw16])

def jacobian(X, w, activation_function):
    '''
    Computes the Jacobian matrix.

    Parameters
    ----------
    X: 500x3 matrix
    w: 16x1 matrix

    Returns
    -------
    matrix
       500x16 matrix of gradients.
    '''
    j = []
    for x in X:
        g = grad(x, w, activation_function)
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

def calculate_b(X, y, w, activation_function):
    '''
    Calculates b.

    Parameters
    ----------
    X: 500x3 matrix
    y: 500x1 matrix
    w: 16x1 matrix
    j: 500x3 jacobian matrix

    Returns
    -------
    np.array
       516x1 matrix
    '''
    r = func(X, w, activation_function) - y
    Dr = jacobian(X, w, activation_function)
    temp = r - np.dot(Dr, w)
    return np.concatenate((temp, np.zeros((16,1))))

def sigmoid(x):
    '''
    Computes the sigmoid function. f(x) = 1/(1+e^-x)

    Parameters
    ----------
    x: The internal value while a pattern goes through the network

    Returns
    -------
    float
       Value after applying sigmoid function
    '''
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    '''
    Computes the derivative of the sigmoid function. f(x) = sigmoid(x)(1 - sigmoid(x))

    Parameters
    ----------
    x: The internal value while a pattern goes through the network

    Returns
    -------
    float
       Value after applying sigmoid function
    '''
    return sigmoid(x) * (1 - sigmoid(x))


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
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

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

def LM(X, y, w, lambda_, gamma, activation_function):
    maxiter = 500
    trust = gamma
    trust_increase = 2
    trust_decrease = 0.8

    loss = 0
    loss_new = 0
    losses = []
    weights = []
    weights.append(w)
    
    for i in range(maxiter):
        Dr = jacobian(X, weights[-1], activation_function)
        Dh = LM_matrix(Dr, lambda_)
        b = calculate_b(X, y, weights[-1], activation_function)
        trust_sqrt = np.sqrt(trust)
        temp = trust_sqrt * weights[-1]
        y_ = np.concatenate((b, temp))
        temp = trust_sqrt * np.identity(16)
        A_ = np.concatenate((Dh, temp))
        pinv = np.linalg.pinv(A_)
        w_new = pinv.dot(y_)
        weights.append(w_new)
        loss_new = calculate_loss(X, y, w_new, lambda_, activation_function)
        losses.append(loss_new)

        if np.linalg.norm(weights[-1] - weights[-2]) < 1e-10:
            break

        if loss_new < loss:
            trust *= trust_decrease
            w = w_new

        else:
            trust *= trust_increase
        
    return losses, weights

def loss_plot(losses, gamma):
    x = range(len(losses))
    y = losses
    plt.plot(x, y)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Initial Gamma={}'.format(gamma))
    plt.show()