# Implementation of the Word Rotator's Distance Score (WRDScore)
# © Mussabayev Ravil, 2024

from typing import List
import numpy as np
import ot
from transformers import AutoTokenizer, AutoModel
import torch
from gensim.models import Word2Vec


class WRDScore:

    def __init__(self, **kwargs):
        if "model" in kwargs.keys():
            self.model_name = kwargs['model']
            if kwargs['model'] == 'codebert':
                # Import CodeBERT model and tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
                self.model = AutoModel.from_pretrained("microsoft/codebert-base")
            elif kwargs['model'] == 'word2vec':
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
                self.default_token = self.tokenizer.tokenize('default')[0]
                self.model = Word2Vec.load('word2vec.model')


    def tokens_to_embeddings(self, tokens, model):
        """
        Convert a sequence of tokens into a numpy array of word2vec embeddings.
        
        Args:
        tokens (list of str): The sequence of tokens.
        model (Word2Vec): The trained Word2Vec model.

        Returns:
        numpy.ndarray: A 2D numpy array where each row corresponds to the embedding of a token.
        """
        embeddings = []
        for token in tokens:
            if token in model.wv:
                embeddings.append(model.wv[token])
            else:
                # Handle out-of-vocabulary tokens
                embeddings.append(model.wv[self.default_token])
        return np.array(embeddings).reshape(len(embeddings), -1)


    def compute_embeddings_and_weights(self, r: List[str], p: List[str]):
        # Obtain CodeBERT embeddings of the reference tokens
        r_tokens = self.tokenizer.tokenize(" ".join(r))
        # print(r_tokens)
        if self.model_name == 'codebert':
            r_token_ids = self.tokenizer.convert_tokens_to_ids(r_tokens)
            r_context_embeddings = self.model(torch.tensor(r_token_ids)[None,:])[0][0]
            r_context_np = r_context_embeddings.detach().numpy()
        else:
            r_context_np = self.tokens_to_embeddings(r_tokens, self.model)

        # Obtain CodeBERT embeddings of the predicted tokens
        p_tokens = self.tokenizer.tokenize(" ".join(p))
        # print(p_tokens)
        if self.model_name == 'codebert':
            p_token_ids = self.tokenizer.convert_tokens_to_ids(p_tokens)
            p_context_embeddings = self.model(torch.tensor(p_token_ids)[None,:])[0][0]
            p_context_np = p_context_embeddings.detach().numpy()
        else:
            p_context_np = self.tokens_to_embeddings(p_tokens, self.model)
        
        # Compute norms for each token in the sequences
        r_norms = np.linalg.norm(r_context_np, axis=1)
        p_norms = np.linalg.norm(p_context_np, axis=1)

        # Compute the weight vectors for the distributions
        r_weights = r_norms / np.sum(r_norms)
        p_weights = p_norms / np.sum(p_norms)

        # Normalize the embeddings by their norms
        r_context_normalized = r_context_np / r_norms[:, np.newaxis]
        p_context_normalized = p_context_np / p_norms[:, np.newaxis]

        return r_context_normalized, p_context_normalized, r_weights, p_weights


    def word_rotator_distance(self, r_context_normalized, p_context_normalized, r_weights, p_weights):
        """
        Compute Word Rotator's Distance between the refence and predicted word sequences.
        :param r: The reference word sequence;
        :param p: The predicted word sequence;
        :return (wrd, T, D): Word Rotator's Distance, coupling (flow) matrix, distance matrix
        """

        # Compute the cosine distance matrix
        # Cosine distance is defined as 1 - cosine similarity
        D = 1 - np.dot(r_context_normalized, p_context_normalized.T)

        # Convert arrays to float64
        D = D.astype(np.float64)
        r_weights = r_weights.astype(np.float64)
        p_weights = p_weights.astype(np.float64)

        # print(D)
        # print(r_weights)
        # print(p_weights)

        wrd = ot.emd2(r_weights, p_weights, D)
        return wrd


    def compute_precision(self, r_context_normalized, p_context_normalized, r_weights, p_weights):
        p_length = p_context_normalized.shape[0]
        precision = 0.
        for j in range(p_length):
            j_dot_products = np.dot(r_context_normalized, p_context_normalized[j])
            precision += p_weights[j] * np.max(j_dot_products)
        return precision


    def wrdscore(self, r: List[str], p: List[str]):
        # Process word embeddings
        r_context_normalized, p_context_normalized, r_weights, p_weights = self.compute_embeddings_and_weights(r, p)
        
        # Compute WRDScore-Recall
        wrd = self.word_rotator_distance(r_context_normalized, p_context_normalized, r_weights, p_weights)
        rc = 1 - wrd

        # Compute WRDScore-Precision
        pr = self.compute_precision(r_context_normalized, p_context_normalized, r_weights, p_weights)

        # Compute WRDScore
        f1 = 2 * pr * rc / (pr + rc)

        return f1


    def bertscore(self, r: List[str], p: List[str]):
        # Process word embeddings
        r_context_normalized, p_context_normalized, r_weights, p_weights = self.compute_embeddings_and_weights(r, p)
        
        # Compute BERTScore-Recall
        rc = self.compute_precision(p_context_normalized, r_context_normalized, p_weights, r_weights)

        # Compute BERTScore-Precision
        pr = self.compute_precision(r_context_normalized, p_context_normalized, r_weights, p_weights)

        # Compute BERTScore
        f1 = 2 * pr * rc / (pr + rc)

        return f1