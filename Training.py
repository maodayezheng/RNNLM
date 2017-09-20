from RNNLM import ContextRNNLM as Model
import sys
import time
import json
import lasagne
import os
import pickle as cPickle
import numpy as np

sys.setrecursionlimit(5000000)
np.set_printoptions(threshold=1000000)
main_dir = sys.argv[0]
out_dir = sys.argv[2]
batch_size = 25
sample_groups = 10
epoch = 4
vocab_size = 500
embed_dim = 128
hid_dim = 256
att_dim = 128
optimizer = lasagne.updates.adam
learning_rate = 1e-4
pre_trained = False
restore_date = "2017_08_30_16_20_53/"
restore_params = "final_model_params.save"
training_data_file = "BPE/train50.tok.bpe.32000.txt"

if __name__ == '__main__':

    # Create model
    model = Model(vocab_size, embed_dim, hid_dim, att_dim)
    if pre_trained:
        with open("code_outputs/" + restore_date + restore_params, "rb") as params:
            print("Params restored from " + restore_date)
            model.set_param_values(cPickle.load(params))
    update_kwargs = {'learning_rate': 1e-4}
    draw_sample = False
    train_step = model.trainingStep(optimizer, learning_rate)
    valid_step = model.validationStep()
    train_data = None

    # Load training and validation data
    with open("SentenceData/" + training_data_file, "r") as dataset:
        train_data = json.loads(dataset.read())

    validation_data = None
    with open("SentenceData/BPE/news2013.tok.bpe.32000.txt", "r") as dev:
        validation_data = json.loads(dev.read())

    validation_data = sorted(validation_data, key=lambda d: max(len(d[0]), len(d[1])))
    len_valid = len(validation_data)
    splits = len_valid % batch_size
    validation_data = validation_data[:-splits]
    validation_data = np.array(validation_data)
    print(" The chosen validation size : " + str(len(validation_data)))
    g = int(len(validation_data) / batch_size)
    print(" The chosen validation groups : " + str(g))
    validation_data = np.split(validation_data, g)

    # split validation data
    validation_pair = []
    for m in validation_data:
        l = len(m[-1])
        start = time.clock()
        source = None
        for datapoint in m:
            s = np.array(datapoint[0])
            if len(s) != l:
                s = np.append(s, [-1] * (l - len(s)))

            if source is None:
                source = s.reshape((1, s.shape[0]))
            else:
                source = np.concatenate([source, s.reshape((1, s.shape[0]))])

        validation_pair.append(source)

    # calculate required iterations
    data_size = len(train_data)
    print(" The training data size : " + str(data_size))
    iters = int(data_size * epoch / (batch_size * sample_groups) + 1)
    print(" The number of iterations : " + str(iters))

    training_loss = []
    validation_loss = []

    for i in range(iters):
        batch_indices = np.random.choice(len(train_data), batch_size * sample_groups, replace=False)
        mini_batch = [train_data[ind] for ind in batch_indices]
        mini_batch = sorted(mini_batch, key=lambda d:len(d))

        mini_batch = np.array(mini_batch)
        mini_batchs = np.split(mini_batch, sample_groups)
        loss = None
        read_attention = None
        write_attention = None
        for m in mini_batchs:
            l = len(m[-1])
            source = None
            start = time.clock()
            for datapoint in m:
                s = np.array(datapoint[0])
                if len(s) != l:
                    s = np.append(s, [-1] * (l - len(s)))
                if source is None:
                    source = s.reshape((1, s.shape[0]))
                else:
                    source = np.concatenate([source, s.reshape((1, s.shape[0]))])

            output = train_step(source)
            iter_time = time.clock() - start
            loss = output[0]
            training_loss.append(loss)

            if i % int(0.1*iters) == 0:
                print("Training time " + str(iter_time) + " sec with sentence length " + str(l))
                print("Total loss ")
                print("RNN loss ")
                print("Selection loss")
                print("Average number of selection ")

        if i % int(0.05*iters) == 0:
            valid_loss = 0
            p = 0
            v_out = None
            for pair in validation_pair:
                p += 1
                v_out = valid_step(pair)
                valid_loss += v_out[0]

            print("The loss on testing set is : " + str(valid_loss / p))
            validation_loss.append(valid_loss / p)

        if i % int(0.1*iters) == 0 and i is not 0:
            print("Params saved at iteration " + str(i))
            np.save(os.path.join(out_dir, 'training_loss.npy'), training_loss)
            np.save(os.path.join(out_dir, 'validation_loss'), validation_loss)
            with open(os.path.join(out_dir, 'model_params.save'), 'wb') as f:
                cPickle.dump(model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()

    np.save(os.path.join(out_dir, 'training_loss.npy'), training_loss)
    np.save(os.path.join(out_dir, 'validation_loss.npy'), validation_loss)
    with open(os.path.join(out_dir, 'final_model_params.save'), 'wb') as f:
        cPickle.dump(model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
    print("Finished training ")