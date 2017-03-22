"""Loss functions."""

import tensorflow as tf
import semver
from keras import backend as K


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    _delta = 1.0
    x1 = 0.5*K.square(y_true - y_pred)
    x2 = _delta*K.abs(y_true - y_pred) - 0.5*(_delta**2)
    threshold = _delta*K.ones(shape = y_true.shape, dtype = y_true.DType)
    condition = K.less_equal(K.abs(y_true - y_pred), threshold)
    return K.where(condition, x1, x2)


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    h_loss = huber_loss(y_true, y_pred)
    return K.reduce_mean(h_loss)
