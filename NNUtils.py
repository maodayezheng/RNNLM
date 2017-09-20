import theano.tensor as T
import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.layers import EmbeddingLayer, InputLayer, DenseLayer
import lasagne
from lasagne.nonlinearities import linear
random = RandomStreams()


class BahdanauAligment(object):
    def __init__(self, input_dim, state_dim, attend_dim):
        v = np.random.uniform(-0.05, 0.05, size=(input_dim, attend_dim))
        self.weight_W = theano.shared(value=v, name="att_weight_W").astype(theano.config.floatX)
        v = np.random.uniform(-0.05, 0.05, size=(state_dim, attend_dim))
        self.weight_U = theano.shared(value=v, name="att_weight_U").astype(theano.config.floatX)
        v = np.zeros((attend_dim, )).astype(theano.config.floatX)
        self.bias = theano.shared(value=v, name="att_bias")
        v = np.random.uniform(-0.05, 0.05, size=(attend_dim,))
        self.weight_v = theano.shared(value=v, name='att_weigth_v').astype(theano.config.floatX)
        self.input_dim = input_dim
        self.attend_dim = attend_dim
        self.state_dim = state_dim

    def __call__(self, x, states):
        b = self.bias
        b = b.reshape((1, 1, self.attend_dim))
        n, d = x.shape
        e = T.dot(x, self.weight_W)
        e = e.reshape((n, 1, self.attend_dim))
        n, l, s = states.shape
        states = states.reshape((n*l, s))
        s = T.dot(states, self.weight_U)
        s = s.reshape((n, l, self.attend_dim))
        content = T.tanh(e + s + b)
        score = T.dot(content.reshape((n*l, self.attend_dim)), self.weight_v)
        score = score.reshape((n, l))
        return score

    def get_params(self):
        return [self.weight_W, self.weight_U, self.weight_v, self.bias]

    def get_params_value(self):
        return [self.weight_W.get_value(), self.weight_U.get_value(), self.weight_v.get_value(), self.bias.get_value()]

    def set_params_value(self, params):
        self.weight_W.set_value(params[0])
        self.weight_U.set_value(params[1])
        self.weight_v.set_value(params[2])
        self.bias.set_value(params[3])


class ContextSelector(object):
    def __init__(self, input_dim, state_dim):
        v = np.random.uniform(-0.05, 0.05, size=(input_dim, state_dim))
        self.weight_W = theano.shared(value=v, name="ads_weight_W").astype(theano.config.floatX)
        v = np.zeros((state_dim, )).astype(theano.config.floatX)
        self.bias = theano.shared(value=v, name="ads_bias")
        self.input_dim = input_dim
        self.state_dim = state_dim

    def __call__(self, x, states):
        content = T.tanh(T.dot(x, self.weight_W) + self.bias.reshape((1, self.state_dim)))
        n, d = content.shape
        content = content.reshape((n, 1, d))
        scores = T.sum(content * states, axis=-1)
        return scores

    def get_params(self):
        return [self.weight_W, self.bias]

    def get_params_value(self):
        return [self.weight_W.get_value(), self.bias.get_value()]

    def set_params_value(self, params):
        self.weight_W.set_value(params[0])
        self.bias.set_value(params[1])


class StraightThrough(theano.Op):
    __props__ = ()

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage, params=None):
        x = inputs[0]
        z = output_storage[0]
        threshold = random.uniform(size=x.shape, low=0.0, high=1.0)
        samples = T.cast(T.gt(x, threshold), "float32")
        z[0] = samples

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        x = inputs[0]
        return [T.ones(shape=x.shape)]


class LM(object):
    def embedding(self, input_dim, cats, output_dim):
        words = np.random.uniform(-0.05, 0.05, (cats, output_dim)).astype("float32")
        w = theano.shared(value=words.astype(theano.config.floatX))
        embed_input = InputLayer((None, input_dim), input_var=T.imatrix())
        e = EmbeddingLayer(embed_input, input_size=cats, output_size=output_dim, W=w)
        return e


class NN(object):
    def mlp(self, input_size, output_size, n_layers=1, activation=linear):
        layer = InputLayer((None, input_size))
        if n_layers > 1:
            for i in range(n_layers - 1):
                layer = DenseLayer(layer, output_size, W=lasagne.init.GlorotUniform(),
                                                  b=lasagne.init.Constant(0.0))
        h = DenseLayer(layer, output_size, nonlinearity=activation, W=lasagne.init.GlorotUniform(),
                       b=lasagne.init.Constant(0.0))

        return h