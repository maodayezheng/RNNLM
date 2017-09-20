import lasagne
import theano.tensor as T
from lasagne.layers import get_output
from lasagne.nonlinearities import sigmoid, tanh

from Models.NNUtils import NN


class GRU(NN):
    def __init__(self, input_dim, hid_dim):
        self.hid_dim = hid_dim
        self.input_dim = input_dim
        self.gate = self.mlp(input_dim, 2 * hid_dim, activation=sigmoid)
        self.candidate = self.mlp(input_dim, hid_dim, activation=tanh)

    def __call__(self, x, h):
        gate_input = T.concatenate([x, h], axis=-1)
        gate = get_output(self.gate, gate_input)
        u = gate[:, :self.hid_dim]
        r = gate[:, self.hid_dim:]
        reset_h = h * r
        c_in = T.concatenate([x, reset_h], axis=1)
        c = get_output(self.candidate, c_in)
        o = (1.0 - u) * h + u * c
        return o

    def get_params(self):
        gate_param = lasagne.layers.get_all_params(self.gate)
        candidate_param = lasagne.layers.get_all_params(self.candidate)

        return gate_param + candidate_param

    def get_param_values(self):
        gate_param = lasagne.layers.get_all_param_values(self.gate)
        candidate_param = lasagne.layers.get_all_param_values(self.candidate)

        return [gate_param, candidate_param]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.gate, params[0])
        lasagne.layers.set_all_param_values(self.candidate, params[1])

