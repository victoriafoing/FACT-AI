from collections import namedtuple
from pathlib import Path
from typing import List, Dict
from utility_functions import *
import config

# Named tuple to store the google analogies in their raw form
RawDatapoint = namedtuple('Raw_Datapoint', ['x1', 'x2', 'x3', 'y', 'task'])

# Named tuple to store the corresponding embeddings of the analogy words
Datapoint = namedtuple('Datapoint', ['analogy_embeddings', 'gt_embedding', 'protected'])

# Function to load the data from the google analogy text file
def load_data(path: Path = Path('./data/google-analogies.txt')) -> List[RawDatapoint]:
    dataset = []
    with path.open() as f:
        for line in f.readlines():
            if line[0] == ':':
                task = line[1:].strip(' \n')
                continue

            words = line.strip('\n').split(' ')
            p = RawDatapoint(*words, task)
            dataset.append(p)

    return dataset

# Function to transform the raw data into their corresponding word embeddings
def transform_data(word_vectors : Dict, analogy_dataset : List[RawDatapoint], use_boluk : bool = False) -> List[Datapoint]:
    # List to store the transformed datapoints
    transformed_dataset = []
    # Obtaining the gender word pairs
    gender_pairs = obtain_gender_pairs(word_vectors)
    # Obtaining the gender subspace
    gender_subspace = obtain_gender_subspace(gender_pairs, use_boluk = use_boluk)
    # For each Raw_Datapoint tuple in the list
    for raw_datapoint in analogy_dataset:
        # Obtaining a list of the corresponding word embeddings
        try:
            embeddings = load_vectors(word_vectors, [raw_datapoint.x1, raw_datapoint.x2, raw_datapoint.x3, raw_datapoint.y])
            embeddings = np.array(embeddings)
        except:
            continue
        # Stacking the embeddings horizontally : [1 X 3D]
        a = np.hstack(tuple(embeddings[0 : 3]))
        # Obtaining the embedding corresponding to y
        b = embeddings[3]
        # Obtaining the embedding corresponding to z (protected variable)
        c = b.dot(gender_subspace.T)
        # Temporary transformed datapoint
        temp_datapoint = Datapoint(*[a, b, c])
        # Adding to the list of transformed datapoints
        transformed_dataset.append(temp_datapoint)
    # Returning the list of transformed datapoints
    return transformed_dataset
        

