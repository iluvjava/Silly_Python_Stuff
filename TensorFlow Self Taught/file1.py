import tensorflow as tf
import numpy as np


def basic():
    # Cannot be reassigned.
    x = tf.constant([1, 2, 3])
    print(x)
    x = tf.Variable(x)  # creates a variable using a constant:
    x[0].assign(-1)  # change one of the entries
    print(x)
    npx = x.numpy()  # convert to numpy array.
    print(npx)
    print(f"type: {type(npx)}")

def AutomatricDiff():
    x = tf.convert_to_tensor([1, 1, 1], dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.tensordot(x, x, 0)
    dy_dx = t.gradient(y, x)
    print(dy_dx)
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.reduce_sum(x*x)
    dy_dx = t.gradient(y, x)
    print(dy_dx)


def main():
    basic()
    AutomatricDiff()

if __name__ ==  "__main__":
    main()
