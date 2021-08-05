import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dropout, InputLayer, Dense

def sigmoid(x):
    return 1/(1 + np.exp(-x))  

def mu_evaluation(model, X, name='discrete', k=None):

    # Default parameter values
    if k is None:
        if name == 'bell':
            k = 4
        elif name == 'tanh':
            k = 5.0
        elif name == 'sigmoid':
            k = 5.0
        elif name == 'discrete':
            k = 0.5
        else:
            raise ValueError("Unknown name, ", name)

    if name == 'bell':
        def sat_func(q):
            q = 1 - np.exp(-k * q**2)
            return q
    elif name == 'tanh':
        def sat_func(q):
            q = np.tanh(k * q)
            # q = np.abs(q)
            return q
    elif name == 'sigmoid':
        def sat_func(q):
            q = sigmoid(k * q)
            return q
    elif name == 'discrete':
        def sat_func(q):
            q = np.abs(q)
            # q = np.array(q)
            q = (q > k)
            q = q.astype(np.float)
            return q

    y = X
    qs = []
    mus = []
    heights = []
    for i, l in enumerate(model.layers):
        z = l(y)
        if isinstance(l, Dense):
            w, b = l.weights
            # print("w: ", w.shape, "y: ", y.shape)
            z_pre_A = tf.einsum("io, ...ni -> ...no", w, y)
            z_pre_B = b
            z_pre = z_pre_A + z_pre_B
            z_2 = l.activation(z_pre)
            assert (np.abs(z_2 - z) < 0.01).all()
            # Measure the saturation of this neuron's input
            q = sat_func(z_pre)            
            # print("q: ", q.shape)
            # mu = np.einsum("no -> n", q) / q.size
            mu = np.mean(q, axis=-1) 
            # print("mu: ", mu.shape)
            #np.mean(q, axis=-1)
            mus.append(mu)
            heights.append(q.shape[-1])
            qs.append(q)
            #print(mu)
        elif isinstance(l, InputLayer):
            pass
        elif isinstance(l, Dropout):
            pass
        else:
            raise ValueError("Unsupported layer " + str(l))
        y = z
    mus = np.stack(mus, axis=0)
    heights = np.stack(heights)
    # print("heights: ", heights)
    # print("mus: ", mus.shape)
    mu_avg = np.mean(mus, axis=0)
    # print("mu_avg: ", mu_avg.shape)
    mu_wavg = np.average(mus, axis=0, weights=heights)
    # print("mu_wavg: ", mu_wavg.shape)
    return mu_avg, mu_wavg, mus


def square():
    N = 100
    M = N
    d = 10
    x_lin = np.linspace(-d, d, N)
    y_lin = np.linspace(-d, d, M)
    x_g, y_g = np.meshgrid(x_lin, y_lin)
    X = np.stack([x_g, y_g], axis=-1)