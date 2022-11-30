import numpy as np
import util

def tanh(x):
    '''
    Compute the tanh function.
    f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Parameters
    ----------
    x: The internal value while a pattern goes through the network.

    Returns
    -------
    float
       Value after applying tanh function.
    '''
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def RMSE(y, t):
    '''
    Compute the root mean square error.
    RMSE = sqrt(Σ(y-t)^2 / N)
    Parameters
    ----------
    y: The predicted values.
    t: The actual values.

    Returns
    -------
    float
       Value after calculating RMSE.
    '''
    return np.sqrt(np.square((y - t)).mean())