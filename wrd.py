# Implementation of the Word Rotator's Distance
# https://github.com/wmayner/pyemd
# © Mussabayev Ravil, 2022

from typing import List
import logging
from gensim.corpora import Dictionary
import numpy as np
from pyemd import emd_with_flow
from numpy import linalg as LA


# Cosine distance
def cos_dis(u, v):
    dist = 1.0 - np.dot(u, v) / (LA.norm(u) * LA.norm(v))
    return dist


# Cosine similarity
def cos_sim(u, v):
    dist = np.dot(u, v) / (LA.norm(u) * LA.norm(v))
    return dist


class WRD:

    def __init__(self, **kwargs):
        if "model_fn" in kwargs.keys():
            self.model = self.load_word_embedding_model(kwargs['model_fn'])
        elif "model" in kwargs.keys():
            self.model = kwargs['model']
        self.words = self.model.keys()
    

    def load_word_embedding_model(self, fn, encoding='utf-8'):
        """
        Return the Word Embedding model at the given path
        :param fn: path where the model of interest is stored
        :param encoding: encoding of the file of interest. Default value is utf-8
        :return:
        """
        logging.info("load_word_embedding_model >>>")
        model = {}
        with open(fn, 'r', encoding=encoding) as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                model[word] = vector
        logging.info("load_word_embedding_model <<<")
        return model


    def nbow(self, document, vocab_len, dictionary):
        d = np.zeros(vocab_len, dtype=np.double)
        nbow = dictionary.doc2bow(document)  # word frequencies
        for idx, freq in nbow:
            d[idx] = freq  # word frequencies
        #nbow = dict(nbow)
        Z = 0.  # normalizing constant
        token2id = dictionary.token2id
        for token in token2id:
            i = token2id[token]
            if token in document:
                Z += d[i] * np.log(LA.norm(self.model[token]))
                d[i] *= np.log(LA.norm(self.model[token]))
                # Z += d[i]
        d /= Z
        return d


    def wrdistance(self, document1: List[str], document2: List[str]):
        """
        Compute Rotator's distance among the two list of documents
        :param document1:
        :param document2:
        :return:
        """

        # Remove out-of-vocabulary words.
        len_pre_oov1 = len(document1)
        len_pre_oov2 = len(document2)
        document1 = [token for token in document1 if token in self.words]
        document2 = [token for token in document2 if token in self.words]
        diff1 = len_pre_oov1 - len(document1)
        diff2 = len_pre_oov2 - len(document2)
        if diff1 > 0 or diff2 > 0:
            logging.info('Removed %d and %d OOV words from document 1 and 2 (respectively).',
                         diff1, diff2)

        if len(document1) == 0 or len(document2) == 0:
            logging.info('At least one of the documents had no words that were'
                         'in the vocabulary. Aborting (returning inf).')
            return float('inf')

        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)

        # Sets for faster look-up.
        docset1 = set(document1)
        docset2 = set(document2)
        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        sim_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        for i, t1 in dictionary.items():
            for j, t2 in dictionary.items():
                if not t1 in docset1 or not t2 in docset2:
                    continue
                # If the current cell is empty compute Euclidean distance between word vectors.
                if not distance_matrix[i, j]:
                    distance_matrix[i, j] = cos_dis(self.model[t1], self.model[t2])  # cosine distance
                    sim_matrix[i, j] = cos_sim(self.model[t1], self.model[t2])  # cosine similarity
                    # Fill the specular cell for saving computation
                    distance_matrix[j, i] = distance_matrix[i, j]
                    sim_matrix[j, i] = sim_matrix[i, j]

        if np.sum(distance_matrix) == 0.0:
            # `emd` gets stuck if the distance matrix contains only zeros.
            logging.info('The distance matrix is all zeros. Aborting (returning inf).')
            return 0.

        # Compute nBOW representation of documents.
        d1 = self.nbow(document1, vocab_len, dictionary)
        d2 = self.nbow(document2, vocab_len, dictionary)

        wmd, T = emd_with_flow(d1, d2, distance_matrix)
        return wmd, T, dictionary, distance_matrix, sim_matrix