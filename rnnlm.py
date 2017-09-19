import theano.tensor as T
import lasagne
from NNUtils import LM, ContextSelector, BahdanauAligment, StraightThrough


class ContextRNN(LM):
    def __init__(self, vocab_size, embed_dim, hid_dim, att_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embed_dim
        self.hid_dim = hid_dim
        self.att_dim = att_dim
        self.selector = ContextSelector(hid_dim, embed_dim)
        self.alignment = BahdanauAligment(hid_dim, embed_dim, att_dim)
        self.sampler = StraightThrough()
        self.input_embedding = self.embedding(vocab_size, vocab_size, embed_dim)
        self.output_embedding = self.embedding(vocab_size, vocab_size, embed_dim)
