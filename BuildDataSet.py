import json
import numpy as np
import os
from nltk.tokenize import word_tokenize

data_directory = "Data/sentence_20k/"
np.set_printoptions(threshold=1000000)
vocab = {}
idx = 0
# Flags
keep_unk = True
# Load vocabulary
unk_count = 0
sentence_idx = []
unique_tokens = []
with open(data_directory + "vocab.txt", "r") as words:
    for line in words:
        vocab[line.rstrip("\n")] = idx
        idx += 1

total = 0

with open(data_directory + "test_sentence.txt", "r") as doc:
        for line in doc:
            print(line)
            total += 1
            s_idx = []
            sentence = word_tokenize(line.lower().rstrip("\n"))
            # only keep sentences who are less than 50 and greater than 5 tokens
            for token in sentence:
                # Replace some rare words with <unk>
                token_idx = vocab.get(token, 1)
                s_idx.append(token_idx)
            s_idx = s_idx + [0]
            sentence_idx.append(s_idx)

with open(data_directory + "test_idx.txt", "w") as doc:
        doc.write(json.dumps(sentence_idx))
