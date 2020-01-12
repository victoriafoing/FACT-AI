import os
from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

def load_pretrained_vectors(data_path, savedir, savefile, glove):
    

    if os.path.isfile(savedir+savefile):
        word_vectors = KeyedVectors.load(savedir+savefile)

    else: 
        if glove:
            tmp_file = get_tmpfile("toword2vec.txt")
            _ = glove2word2vec(data_path, tmp_file)
            data_path = tmp_file

        word_vectors = KeyedVectors.load_word2vec_format(data_path, binary=False)
        word_vectors.save(savedir+savefile)

    return word_vectors


def get_words_from_file(datapath, word_list):
    with open(datapath+word_list, "r") as f:
        words = f.read().split()
    return list(set(words))


def load_vectors(data_path, savedir, savefile, data_point, glove=False):
    # data_point is a list of words
    if not os.path.exists(f'{savedir}'):
        os.makedirs(f'{savedir}')

    word_vectors = load_pretrained_vectors(data_path, savedir, savefile, glove)
    #words = get_words_from_file(config.data_path, config.word_list)
    vectors = word_vectors[data_point]
    return vectors

# Example: 
# Wikipedia
#vectors = load_vectors("data/enwiki_20180420_win10_100d.txt/data", "pre-trained/", "wikipedia", ['athens', 'greece'])

# GloVe
#vectors = load_vectors("data/glove.twitter.27B.100d.txt", "pre-trained/", "glove", ['athens', 'greece'], True)
#print(vectors)
