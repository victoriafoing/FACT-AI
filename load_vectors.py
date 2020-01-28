import os
from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from typing import Dict, List
import utility_functions
import gzip

def load_pretrained_vectors(data_path, savedir, embedding_type):

    # If the save directory does not exist
    if not os.path.exists(f'{savedir}'):
        os.makedirs(f'{savedir}')

    if os.path.isfile(savedir + embedding_type):
        print("Loading from saved file.")
        word_vectors = KeyedVectors.load(savedir + embedding_type)
    else:
        if embedding_type == 'google':
            with gzip.GzipFile(fileobj=open(data_path, "rb", buffering=0)) as f:
                word_vectors = utility_functions.load_word2vec_format(f, max_num_words=2000000)
        else:
            if embedding_type == 'glove':
                tmp_file = get_tmpfile("toword2vec.txt")
                _ = glove2word2vec(data_path, tmp_file)
                data_path = tmp_file
            word_vectors = KeyedVectors.load_word2vec_format(data_path, binary=False)

        word_vectors.save(savedir + embedding_type)

    return word_vectors


def get_words_from_file(datapath, word_list):
    with open(datapath+word_list, "r") as f:
        words = f.read().split()
    return list(set(words))


def load_vectors(word_vectors : Dict, data_point : List) -> List[List]:
    #words = get_words_from_file(config.data_path, config.word_list)
    data_point = [x.lower() for x in data_point]
    vectors = word_vectors[data_point]
    return vectors.tolist()

# Example: 
# Wikipedia
#vectors = load_vectors("data/enwiki_20180420_win10_100d.txt/data", "pre-trained/", "wikipedia", ['athens', 'greece'])

# GloVe
#vectors = load_vectors("data/glove.twitter.27B.100d.txt", "pre-trained/", "glove", ['athens', 'greece'], True)
#print(vectors)
