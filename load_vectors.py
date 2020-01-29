import os
from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from typing import Dict, List
import gzip
import wget


def load_pretrained_vectors(embedding_type):

    wiki_data_path = "data/enwiki_20180420_300d.txt"
    google_data_path = "data/GoogleNews-vectors-negative300.bin.gz"
    glove_data_path = "data/glove.42B.300d.txt"
    
    # Load saved vectors from save directory.
    if os.path.isfile('data/'+embedding_type+'_pre-trained'):
        print("Loading from saved file.")
        word_vectors = KeyedVectors.load('data/'+embedding_type+"_pre-trained")

    else:
        # Download and save vectors
        if embedding_type == 'GoogleNews':
            if not os.path.isfile(google_data_path):
                  wget.download("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", out="data/")
                
            with gzip.GzipFile(fileobj=open(google_data_path, "rb", buffering=0)) as f:
                    word_vectors = KeyedVectors.load_word2vec_format(google_data_path, binary=True) 

        elif embedding_type == 'Glove':
            if not os.path.isfile(glove_data_path):
                wget.download("http://nlp.stanford.edu/data/glove.42B.300d.zip", out="data/")

            with gzip.GzipFile(fileobj=open(glove_data_path, "rb", buffering=0)) as f:
                tmp_file = get_tmpfile("toword2vec.txt")
                _ = glove2word2vec(glove_data_path, tmp_file)
                data_path = tmp_file
                word_vectors = KeyedVectors.load_word2vec_format(glove_data_path, binary=False)

        elif embedding_type == 'Wikipedia2Vec':
            if not os.path.isfile(wiki_data_path):
                wget.download("http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.txt.bz2", out="data/")

            word_vectors = KeyedVectors.load_word2vec_format(wiki_data_path, binary=False)

        word_vectors.save('data/'+embedding_type+'_pre-trained')

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
#load_pretrained_vectors("Glove")
# GloVe
#vectors = load_vectors("data/glove.twitter.27B.100d.txt", "pre-trained/", "glove", ['athens', 'greece'], True)
#print(vectors)
