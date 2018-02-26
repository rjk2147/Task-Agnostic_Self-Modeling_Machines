import tensorflow as tf
import numpy as np


def log10(t):
    """
    Calculates the base-10 log of each element in t.
    @param t: The tensor from which to calculate the base-10 log.
    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = tf.log(t)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def loss_p(y, y_hat, p=2):
    return tf.reduce_mean(tf.abs(y - y_hat) ** p)


# def loss_bce(y, y_hat):
#    return -tf.reduce_mean(y_hat * tf.log(y) + (1. - y_hat) * tf.log(1. - y))
def loss_bce_alt(y, y_hat):
    t = type(y_hat)
    l = tf.ones_like(y)
    if t is tf.Tensor:
        mul = tf.concat([y_hat, (1 - y_hat)], axis=1)
    else:
        mul = np.array([y_hat, 1 - y_hat])
    l = l * mul
    return tf.squeeze(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=l)))


def loss_bce(preds, targets):
    """
    Calculates the sum of binary cross-entropy losses between predictions and ground truths.
    @param preds: A 1xN tensor. The predicted classifications of each frame.
    @param targets: A 1xN tensor The target labels for each frame. (Either 1 or -1). Not "truths"
                    because the generator passes in lies to determine how well it confuses the
                    discriminator.
    @return: The sum of binary cross-entropy losses.
    """
    return tf.squeeze(-1 * (tf.matmul(targets, log10(preds), transpose_a=True) +
                            tf.matmul(1 - targets, log10(1 - preds), transpose_a=True)))


def loss_smooth(y, y_last, p=2):
    return tf.norm(y - y_last, ord=p)