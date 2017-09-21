import lasagne
import theano
import theano.tensor as T
from Models.RNNCells import GRU
from lasagne.layers import get_output
from lasagne.nonlinearities import tanh
from theano import scan, function
from theano.gradient import zero_grad
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from Models.NNUtils import NN, LM, ContextSelector, BahdanauAligment

random = RandomStreams()


class ContextRNNLM(NN, LM):
    def __init__(self, vocab_size, embed_dim, hid_dim, att_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embed_dim
        self.hid_dim = hid_dim
        self.att_dim = att_dim

        self.input_embedding = self.embedding(vocab_size, vocab_size, embed_dim)
        self.output_embedding = self.embedding(vocab_size, vocab_size, embed_dim)

        self.selector = ContextSelector(hid_dim, embed_dim)
        self.alignment = BahdanauAligment(hid_dim, embed_dim, att_dim)
        self.cell = GRU(2*embed_dim+hid_dim, hid_dim)

        self.content_encoder = self.mlp(embed_dim, hid_dim, activation=tanh)
        self.content_compressor = self.mlp(hid_dim, embed_dim, activation=tanh)
        self.output_layer = self.mlp(2*embed_dim + hid_dim, embed_dim, activation=tanh)

    def rnnStep(self, t, x, h, selection, candidates, output_embed, mask):
        # Compute the Context feed into RNN
        n, l, d = candidates.shape
        # Compute the alignment score
        alignment_scores = self.alignment(h, candidates)
        alignment_scores = T.exp(alignment_scores)
        alignment_scores = theano.printing.Print(" The alignment score 2 ")(alignment_scores)
        valid_alignments = selection * mask * alignment_scores[:, :-1]
        valid_alignments = theano.printing.Print(" The valid score 1 ")(valid_alignments)

        s = alignment_scores[:, -1]
        normalizor = T.sum(valid_alignments, axis=-1) + s
        valid_alignments = T.concatenate([valid_alignments, s.reshape((n, 1))], axis=-1)
        valid_score = T.true_div(valid_alignments, normalizor.reshape((n, 1)))
        valid_score = theano.printing.Print(" The valid score 2 ")(valid_score)
        context = T.sum(valid_score.reshape((n, l, 1)) * candidates, axis=1)

        # RNN computation
        rnn_in = T.concatenate([x, context], axis=-1)
        h_next = self.cell(rnn_in, h)
        output = T.concatenate([x, context, h], axis=-1)
        output = get_output(self.output_layer, output)

        # Get the word for next time step
        scores = T.dot(output, output_embed)
        greedy_predictions = zero_grad(T.argmax(scores, axis=-1, keepdims=True))
        x_next = get_output(self.input_embedding, greedy_predictions)
        return x_next, h_next, output, scores, greedy_predictions

    def forward(self, source, encode_mask):
        # Create input mask
        input_embedding = get_output(self.input_embedding, source)
        n, l = encode_mask.shape

        # Compute the content of input
        content = T.sum(encode_mask.reshape((n, l, 1)) * input_embedding, axis=1)

        # Compute the selected words
        content = get_output(self.content_encoder, content)
        context_score = self.selector(content, input_embedding)
        selective_probs = T.nnet.sigmoid(context_score)

        # Convert to binary
        threshold = random.uniform(size=selective_probs.shape, low=0.0, high=1.0)
        selection = T.cast(T.gt(selective_probs + threshold, 1), "float32")
        # Init the RNN
        h_init = content
        start_init = T.zeros((n, self.embedding_dim), dtype="float32")
        dense_content = get_output(self.content_compressor, content)
        time_step = T.arange(l)
        candidates = T.concatenate([input_embedding, dense_content.reshape((n, 1, self.embedding_dim))], axis=1)
        out_embed = self.output_embedding.W
        ([x, hidden, o, scores, greedy_preds], update) = scan(self.rnnStep,
                                                              outputs_info=[start_init, h_init, None, None, None],
                                                              sequences=[time_step],
                                                              non_sequences=[selection, candidates,
                                                                             out_embed.T, encode_mask])

        return selective_probs, selection, o, scores, greedy_preds

    def loss(self):
        source = T.imatrix('source')
        encode_mask = T.cast(T.ge(source, 0), "float32")
        decode_mask = T.cast(T.gt(source, -1), "float32")

        # Get the forward results from model
        selective_probs, selection, o, output_scores, greedy_preds = self.forward(source, encode_mask)

        l, n, k = output_scores.shape
        # Compute the loss on output sequence (per word)
        output_embed = get_output(self.output_embedding, source)
        decode_mask = decode_mask.dimshuffle((1, 0))
        output_embed = output_embed.dimshuffle((1, 0, 2))
        true_target_score = T.sum(output_embed * o, axis=-1)
        clip = zero_grad(T.max(output_scores, axis=-1))
        true_target_score = T.exp(true_target_score - clip)
        output_scores = T.exp(output_scores - clip.reshape((l, n, 1)))
        normalizor = T.sum(output_scores, axis=-1)
        output_probs = T.true_div(true_target_score, normalizor)
        num_words = T.sum(encode_mask, axis=-1)
        rnn_loss = T.mean(T.sum(-T.log(output_probs + 1.0e-5) * decode_mask, axis=0) / num_words)

        # Compute the loss on selection (per selection)
        valid_selection = zero_grad(selection * encode_mask)
        num_selection = T.sum(valid_selection, axis=-1)
        selection_loss = T.mean(T.sum(-T.log(1.0 - selective_probs) * valid_selection, axis=1) / num_selection)
        average_selection = T.mean(valid_selection)
        # total loss
        total_loss = rnn_loss + 0.1*selection_loss

        return source, total_loss, rnn_loss, selection_loss, greedy_preds, average_selection

    def trainingStep(self, optimizer, learning_rate):
        update_kwargs = {}
        source, total_loss, rnn_loss, selection_loss, greedy_preds, average_selection = self.loss()
        params = self.get_params()
        grads = T.grad(total_loss, params)
        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = params
        update_kwargs['learning_rate'] = learning_rate

        updates = optimizer(**update_kwargs)

        training_step = function(inputs=[source],
                                 outputs=[total_loss, rnn_loss, selection_loss, average_selection, greedy_preds],
                                 updates=updates,
                                 allow_input_downcast=True)
        return training_step

    def validationStep(self):
        source, total_loss, rnn_loss, selection_loss, greedy_preds, average_selection = self.loss()
        validation_step = function(inputs=[source],
                                   outputs=[total_loss, rnn_loss, selection_loss, average_selection, greedy_preds],
                                   allow_input_downcast=True)

        return validation_step

    def get_params(self):
        params = lasagne.layers.get_all_params(self.input_embedding)
        params += lasagne.layers.get_all_params(self.output_embedding)
        params += self.selector.get_params()
        params += self.alignment.get_params()
        params += self.cell.get_params()
        params += lasagne.layers.get_all_params(self.content_encoder)
        params += lasagne.layers.get_all_params(self.content_compressor)
        params += lasagne.layers.get_all_params(self.output_layer)

        return params

    def get_param_values(self):
        values = []
        values.append(lasagne.layers.get_all_param_values(self.input_embedding))
        values.append(lasagne.layers.get_all_param_values(self.output_embedding))
        values.append(self.selector.get_params_value())
        values.append(self.alignment.get_params_value())
        values.append(self.cell.get_param_values())
        values.append(lasagne.layers.get_all_param_values(self.content_encoder))
        values.append(lasagne.layers.get_all_param_values(self.content_compressor))
        values.append(lasagne.layers.get_all_param_values(self.output_layer))

        return values

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.output_embedding, params[1])
        self.selector.set_params_value(params[2])
        self.alignment.set_params_value(params[3])
        self.cell.set_param_values(params[4])
        lasagne.layers.set_all_param_values(self.content_encoder, params[5])
        lasagne.layers.set_all_param_values(self.content_compressor, params[6])
        lasagne.layers.set_all_param_values(self.output_layer, params[7])



