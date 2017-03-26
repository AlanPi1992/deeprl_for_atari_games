"""Loss functions."""

import tensorflow as tf
import semver


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
    error = y_true - y_pred
    try:
        loss = tf.select(tf.abs(error)<max_grad, 0.5*tf.square(error), tf.abs(error)-0.5)
    except:
        loss = tf.where(tf.abs(error)<max_grad, 0.5*tf.square(error), tf.abs(error)-0.5)
    return tf.reduce_mean(loss)


def mean_huber_loss_duel(y_true, y_pred, max_grad=10.):
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
    error = y_true - y_pred
    try:
        loss = tf.select(tf.abs(error)<max_grad, 0.5*tf.square(error), 10*tf.abs(error)-50)
    except:
        loss = tf.where(tf.abs(error)<max_grad, 0.5*tf.square(error), 10*tf.abs(error)-50)
    return tf.reduce_mean(loss)