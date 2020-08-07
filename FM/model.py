import tensorflow as tf
import numpy as np


class FactorizationMachine:
    def __init__(self,
                 data_shape: int = 2,
                 latent_shape: int = 5):
        """
        Initialise coefficients.

        For now:
         - Requires manual specification of feature space dimensionality.
         - Coefficients are all float64. Might be overkill but it simplifies casting.

        :param m: Number of features.
        :param k: Number of latent factors to model in V.
        """
        self.b = tf.Variable(tf.zeros([1],
                                      dtype="float32"))
        self.w = tf.Variable(tf.random.normal(([data_shape]),
                                              stddev=0.01,
                                              dtype="float32"))
        self.v = tf.Variable(tf.random.normal(([latent_shape, data_shape]),
                                              stddev=0.01,
                                              dtype="float32"))

    def __call__(self, x: tf.Tensor):
        """
        Predict from model.

        :param x: Tensor containing features.
        :return: Tensor containing predictions.
        """

        # Linear terms
        linear = tf.reduce_sum(tf.multiply(self.w, x),
                               axis=1,
                               keepdims=True)

        # Interaction terms
        interactions = tf.multiply(0.5, tf.reduce_sum(tf.subtract(tf.pow(tf.matmul(x, tf.transpose(self.v)), 2),
                                                                  tf.matmul(tf.pow(x, 2),
                                                                            tf.transpose(tf.pow(self.v, 2)))),
                                                      axis=1,
                                                      keepdims=True))

        # Linear sum along with intercept
        wv = tf.add(linear, interactions)
        bwv = tf.add(self.b, wv)

        return bwv