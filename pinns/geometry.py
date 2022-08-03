import tensorflow as tf


def domain(minval, maxval, num_domain):
    domain_space = tf.random_uniform_initializer(minval, maxval)
    domain_points = tf.Variable(domain_space(shape=[num_domain, 1]), dtype=tf.float32)
    return domain_points
