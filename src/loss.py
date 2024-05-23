from typing import Tuple

import numpy as np
import tensorflow as tf
from scipy.optimize import linprog


def distance_matrix(y_true, y_pred):
    """Returns the euclidean distance matrix.

    Parameters
    ----------
    y_true : [n,dim] matrix
        ground-truth source positions
    y_pred : [m,dim] matrix
        candidate source positions

    Returns
    -------
    [m,n] matrix
        euclidean distance matrix
    """
    yp_sq_norm = tf.reshape(tf.reduce_sum(tf.square(y_pred), 0), [-1, 1]) # reshape as rows
    yt_sq_norm = tf.reshape(tf.reduce_sum(tf.square(y_true), 0), [1, -1]) # reshape as columns
    return tf.sqrt(tf.maximum(yp_sq_norm + yt_sq_norm - 2*tf.matmul(y_pred, y_true, True, False) ,1e-32))


def _setup_positivity_constraint(I,J):
    """Constructs the positivity constraints for the constrained k-means optimization problem.

    The constraints ensure that the elements of the T matrix are non-negative.

    Args:
    ----
        I: The number of candidates.
        J: The number of ground-truth labels.

    Returns:
    -------
        A tuple containing the A and b matrices for the positivity constraints, where A is a matrix of shape (I*J, I*J)
        and b is a vector of length I*J.
    """
    return (-1*np.eye(I*J) , np.zeros(I*J))

def _setup_tau_constraint(I: int, J: int, tau: int) -> Tuple[np.ndarray, np.ndarray]:
    """Constructs the tau constraints for the constrained k-means optimization problem.

    The constraint ensures that the rows of the T matrix sum up to -1.

    Args:
    ----
        I: The number of candidates.
        J: The number of ground-truth labels.
        tau: The minimum number of candidates that must be assigned to each ground-truth label.

    Returns:
    -------
        A tuple containing the A and b matrices for the tau constraints, where A is a matrix of shape (J, I*J)
        and b is a vector of length J.
    """
    A = np.zeros((J,I,J))
    A[np.arange(J),:,np.arange(J)] = -1
    b = np.zeros(J) - tau
    return (A.reshape((J,-1)) , b)


def _equality_constraint(I: int, J: int) -> Tuple[np.ndarray, np.ndarray]:
    """Constructs the equality constraints for the constrained k-means optimization problem.

    The constraint ensures that the rows of the T matrix sum up to -1.

    Args:
    ----
        I: The number of candidates.
        J: The number of ground-truth labels.

    Returns:
    -------
        A tuple containing the A and b matrices for the equality constraints, where A is a matrix of shape (I, I*J)
        and b is a vector of length I.
    """
    A = np.zeros((I,I,J))
    A[np.arange(I),np.arange(I),:] = -1
    b = np.ones(I)*-1
    return (A.reshape((I,-1)), b)


def constrained_kmeans(Rho,tau):
    """Implements the constrained k-means algorithm.

    It takes two inputs:

    Rho:
        a distance matrix of shape (I, J).
        I specifies the number of candidates and J represents the number of ground-truth labels.
    tau:
        a scalar value representing the minimum number of candidates that must be assigned to each
        ground-truth label.

    The function returns a matrix of the same shape as Rho, where each element is 0 or 1.
    1 indicates a valid assignment, and 0 indicates an invalid assignment.
    """
    I, J = Rho.shape
    ### inequality constraints ###
    A_pos, b_pos = _setup_positivity_constraint(I,J)
    A_tau, b_tau = _setup_tau_constraint(I,J,tau)
    ### equality constraints ###
    A_eq, b_eq = _equality_constraint(I, J)
    #solve
    T = linprog(Rho.flatten(),
                A_ub=np.concatenate([A_tau, A_pos]), b_ub=np.concatenate([b_tau,b_pos]),
                A_eq=A_eq,b_eq=b_eq,method='highs')
    return np.float32(T.x.reshape(Rho.shape))


@tf.function(
    input_signature=[tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16)])
def tf_constrained_kmeans(Rho,tau):
    """Applies the constrained k-means algorithm to a distance matrix.

    Args:
    ----
        Rho:
            A distance matrix of shape (I, J), where I specifies the number of candidates
            and J represents the number of ground-truth labels.
        tau:
            A scalar value representing the minimum number of candidates that must be assigned
            to each ground-truth label.

    Returns:
    -------
        A matrix of the same shape as Rho, where each element is 0 or 1.
        1 indicates a valid assignment, and 0 indicates an
        invalid assignment.
    """
    return tf.numpy_function(func=constrained_kmeans,inp=[Rho,tau],Tout=(tf.float32), stateful=False)


def remove_zero_padding(A):
    """Removes zero padding from a 2D array or tensor.

    Args:
    ----
        A: A 2D array or tensor with zero padding.

    Returns:
    -------
        A tensor containing the same data as the input tensor, but without the zero padding.
    """
    A = tf.convert_to_tensor(A)
    non_zero_columns = tf.reduce_any(tf.math.not_equal(A,0),axis=0)
    A = tf.boolean_mask(A,non_zero_columns,axis=1)
    non_zero_rows = tf.reduce_any(tf.math.not_equal(A,0),axis=1)
    A = tf.boolean_mask(A,non_zero_rows,axis=0)
    return A
