import mxnet as mx


class BahdanauAligment(object):
    def __init__(self, batch_size, input_dim, state_dim, attend_dim):
        self.e_weight_W = mx.sym.Variable('energy_W_weight', shape=(input_dim, state_dim))
        self.e_weight_U = mx.sym.Variable('energy_U_weight', shape=(state_dim, attend_dim))
        self.e_weight_v = mx.sym.Variable('energy_v_bias', shape=(attend_dim, 1))
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.attend_dim = attend_dim
        self.state_dim = state_dim

    def alignment_score(self, x, states):


