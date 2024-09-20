import json
import collections
import argparse
import random
import numpy as np
from collections import Counter
from collections import defaultdict
from nltk.corpus import reuters
from numpy.linalg import svd


from util import *

random.seed(42)

def extract_unigram_features(ex):

    bow_features = {}
    
    # Combine the premise (sentence1) and hypothesis (sentence2)
    combined_sentences = ex['sentence1'] + ex['sentence2']
    
    # Iterate through each word in the combined list
    for word in combined_sentences:
        # Increment the word count in the bow_features dictionary
        if word in bow_features:
            bow_features[word] += 1
        else:
            bow_features[word] = 1
    
    return bow_features

def extract_custom_features(ex):

    bow_dict = collections.defaultdict(int)
    
    premise = ' '.join(ex["sentence1"])
    hypothesis = ' '.join(ex["sentence2"])
    
    for n in range(1, 5):
        words = premise.split()
        premise_ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

        words2 = hypothesis.split()
        hypothesis_ngrams = [tuple(words2[i:i+n]) for i in range(len(words2)-n+1)]

        for ngram in premise_ngrams + hypothesis_ngrams:
            ngram_str = ' '.join(ngram)
            bow_dict[ngram_str] += 1
    
    return dict(bow_dict)

def learn_predictor(train_examples, valid_examples, feature_extractor, learning_rate=0.01, num_iters=100):

    vocab_set = set()
    for ex in train_examples + valid_examples:
      vocab_set.update(feature_extractor(ex).keys())
    vocab = {word: i for i, word in enumerate(vocab_set)}

    w = np.zeros(len(vocab))

    for it in range(num_iters):
        total_loss = 0
        total_error = 0
        for ex in train_examples:

            features = feature_extractor(ex)
            
            feature_vector = np.zeros(len(vocab))
            for word, count in features.items():
              if word in vocab:
                feature_vector[vocab[word]] = count
                
            x = feature_vector

            z = np.dot(w, x)
            prediction = 1 / (1 + np.exp(-z))
            y = ex["gold_label"]

            gradient = (prediction - y) * x
            w -= learning_rate * gradient


            loss = - (y * np.log(prediction) + (1 - y) * np.log(1 - prediction))
            total_loss += loss


            predicted_label = 1 if prediction >= 0.5 else 0
            if predicted_label != y:
                total_error += 1


        error_rate = total_error / len(train_examples)
        print(f"Iteration {it+1}/{num_iters} - Loss: {total_loss:.4f}, Training Error Rate: {error_rate:.4f}")

    weights = {word: w[i] for word, i in vocab.items()}

    return weights



def count_cooccur_matrix(tokens, window_size=4):
    word2ind = {}
    ind2word = []
    vocab_size = 0
    
    for token in tokens:
        if token not in word2ind:
            word2ind[token] = vocab_size
            ind2word.append(token)
            vocab_size += 1

    co_mat = np.zeros((vocab_size, vocab_size))

    for i, word in enumerate(tokens):
        word_index = word2ind[word]
        left_window = max(0, i - window_size)
        right_window = min(len(tokens), i + window_size + 1)

        for j in range(left_window, right_window):
            if i != j:
                neighbor_index = word2ind[tokens[j]]
                co_mat[word_index][neighbor_index] += 1

    return word2ind, co_mat


def cooccur_to_embedding(co_mat, k=100):
    """
    Perform truncated SVD on the co-occurrence matrix and return the reduced word vectors.
    
    Parameters:
        co_mat : np.array
            Co-occurrence matrix
        k : int
            Number of dimensions to keep after truncation (top k singular values)
    
    Returns:
        U_reduced : np.array
            Reduced word vectors (U * Sigma for the top k singular values)
    """
    # Step 1: Perform full SVD
    U, Sigma, Vt = np.linalg.svd(co_mat, full_matrices=False)
    
    # Step 2: Truncate to the top k singular values
    U_reduced = U[:, :k]      # Take the first k columns of U
    Sigma_reduced = np.diag(Sigma[:k])  # Take the top k singular values
    
    # Step 3: Return the reduced word vectors U * Sigma
    U_reduced = np.dot(U_reduced, Sigma_reduced)
    
    return U_reduced



def top_k_similar(word_ind, embeddings, word2ind, k=10, metric='dot'):
    
    similarities = []
    
    if metric == 'dot':
        for i in range(embeddings.shape[0]):
            if i != word_ind:
                sim = np.dot(embeddings[word_ind], embeddings[i])
                similarities.append((i, sim))

    elif metric == 'cosine':
        word_vector = embeddings[word_ind]
        word_norm = np.linalg.norm(word_vector)

        for i in range(embeddings.shape[0]):
            if i != word_ind:
                neighbor_vector = embeddings[i]
                sim = np.dot(word_vector, neighbor_vector) / (word_norm * np.linalg.norm(neighbor_vector))
                similarities.append((i, sim))

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_k_indices = [index for index, _ in similarities[:k]]

    ind2word = {index: word for word, index in word2ind.items()}
    topk_words = [ind2word[i] for i in top_k_indices]

    return topk_words
