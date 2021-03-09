import numpy as np
import math


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    print("\nSoftmax Result: " + str(p))

    return p


def positional_encoding(mat):
    '''
    :param mat: [T, dim] shaped 2D array
    :return: Positional Encoding Output [T,dim] 2D array
    '''
    T, dim = np.shape(mat)
    pe = np.zeros([T, dim])
    position = np.array([s for s in range(0, T)]).reshape([-1, 1])
    div_term = np.exp(
        np.array([s for s in range(0, dim, 2)]) * -(math.log(10000.0) / dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    print("\nPositional Encoding: " + str(pe))
    return pe


def attention(query, key, value):
    '''
    Compute 'Scaled Dot Product Attention
    :param query: [T,dim] 2D array
    :param key: [T,dim] 2D array
    :param value: [T,dim] 2D array
    :return: self attention output [T,dim] 2D array
    '''

    d_k = np.shape(query)[-1]
    scores = np.matmul(query, np.transpose(key)) / math.sqrt(d_k)

    p_attn = softmax(scores, axis=1)

    print("\nAttention: " + str(np.matmul(p_attn, value)))

    return np.matmul(p_attn, value)


def cosine_similarity(vector1, vector2):
    cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1,
                                                        ord=None) * np.linalg.norm(vector2, ord=None))
    print("\nCosine score: " + cosine)
    return cosine


def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))
