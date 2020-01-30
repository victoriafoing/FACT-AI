import os
from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from typing import Dict, List
import gzip
import wget
import zipfile


def load_pretrained_vectors(embedding_type):
    assert embedding_type in ("GoogleNews", "Glove", "Wikipedia2Vec"), "Invalid embedding type selected."


    wiki_data_path = os.path.join("data", "enwiki_20180420_300d.txt.bz2")
    google_data_path = os.path.join("data", "GoogleNews-vectors-negative300.bin.gz")
    glove_data_path = os.path.join("data", "glove.42B.300d.zip")
    
    # Load saved vectors from save directory.
    if os.path.isfile(os.path.join('data', embedding_type + '_pre-trained')):
        print("Loading from saved file.")
        word_vectors = KeyedVectors.load(os.path.join('data', embedding_type + "_pre-trained"))

    else:
        # Download and save vectors
        if embedding_type == 'GoogleNews':
            if not os.path.isfile(google_data_path):
                print("Downloading vectors..")
                wget.download("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", out="data")
                
            with gzip.GzipFile(fileobj=open(google_data_path, "rb", buffering=0)) as f:
                    word_vectors = KeyedVectors.load_word2vec_format(google_data_path, binary=True, limit = 2000000) 

        elif embedding_type == 'Glove':
            if not os.path.isfile(glove_data_path):
                print("Downloading vectors..")
                wget.download("http://nlp.stanford.edu/data/glove.42B.300d.zip", out="data")
            
            glove_temp_path = glove_data_path.split('.')[ : -1]
            glove_temp_path = '.'.join(glove_temp_path + ['txt'])

            if not os.path.isfile(glove_temp_path):
                with zipfile.ZipFile(glove_data_path, 'r') as f:
                    f.extractall('data')
            
            tmp_file = get_tmpfile("toword2vec.txt")
            _ = glove2word2vec(glove_temp_path, tmp_file)
            data_path = tmp_file
            word_vectors = KeyedVectors.load_word2vec_format(data_path, binary = False, limit = 2000000)

            # with gzip.GzipFile(fileobj=open(glove_data_path, "rb", buffering=0)) as f:
            #     tmp_file = get_tmpfile("toword2vec.txt")
            #     _ = glove2word2vec(glove_data_path, tmp_file)
            #     data_path = tmp_file
            #     word_vectors = KeyedVectors.load_word2vec_format(data_path, binary = False)

        elif embedding_type == 'Wikipedia2Vec':
            if not os.path.isfile(wiki_data_path):
                print("Downloading vectors..")
                wget.download("http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.txt.bz2", out="data")

            with gzip.GzipFile(fileobj=open(wiki_data_path, "rb", buffering=0)) as f:
                print("Retrieving vectors..")
                word_vectors = KeyedVectors.load_word2vec_format(wiki_data_path, binary=False, limit = 2000000)

        word_vectors.save(os.path.join('data', embedding_type + '_pre-trained'))

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


#load_pretrained_vectors("Wikipedia2Vec")