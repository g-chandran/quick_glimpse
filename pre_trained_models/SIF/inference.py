import os, sys
from pathlib import Path

artifact_path = os.getcwd() +  '/pre_trained_models/'
sys.path.append(artifact_path + 'SIF/')
import data_io, params, SIF_embedding


def load_model():
    wordfile = "glove path (glove.840B.300d.txt file)" # you can download glove from https://www.kaggle.com/takuok/glove840b300dtxt
    weightfile = artifact_path + '/SIF/enwiki_vocab_min200.txt'  # each line is a word and its frequency
    weightpara = 1e-3  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    rmpc = 1  # number of principal components to remove in SIF weighting scheme

    (words, We) = data_io.getWordmap(wordfile)

    a = list(words.keys())
    for i,v in enumerate(a):
        words[v.decode("utf-8")] = words.pop(v)

    # load word weights
    word2weight = data_io.getWordWeight(weightfile, weightpara)  # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word

    return (words, weight4ind, rmpc, We)


def generate_vecs(models, document):
    words, weight4ind, rmpc, We = models

    x, m = data_io.sentences2idx(document, words)
    # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    w = data_io.seq2weight(x, m, weight4ind)  # get word weights

    # set parameters
    param = params.params()
    param.rmpc = rmpc
    # get SIF embedding
    embedding = SIF_embedding.SIF_embedding(We, x, w, param)  # embedding[i,:] is the embedding for sentence i
    return embedding