import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    #Your code here
    x = np.random.random([n,1])
    return x
    #raise NotImplementedError

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    A = np.random.random([h,w])
    B = np.random.random([h,w])
    s = A + B
    return [A,B,s]
    #Your code here
    #raise NotImplementedError


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    s = np.linalg.norm(A + B)
    return s
    #Your code here
    #raise NotImplementedError


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    x = np.matmul(inputs.transose(),weights)
    x = np.tanh(x)
    return x
    #Your code here
    #raise NotImplementedError

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    if x <= y:
      return x * y
    else:
      return x/y
    #Your code here
    #raise NotImplementedError

def vector_function(x, y):
    vect = np.vectorize(scalar_function)
    return vect(x,y)
    """
    Make sure vector_function can deal with vector input x,y
    """
    #Your code here
    #raise NotImplementedError
