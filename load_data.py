from collections import namedtuple
from pathlib import Path
from typing import List
from utility_functions import obtain_gender_subspace, obtain_vector_projection

# Named tuple to store the google analogies in their raw form
Raw_Datapoint = namedtuple('Raw_Datapoint', ['x1', 'x2', 'x3', 'y', 'task'])

# Named tuple to store the corresponding embeddings of the analogy words
Datapoint = namedtuple('Datapoint', ['analogy_embeddings', 'gt_embedding', 'protected_embedding'])

# Function to load the data from the google analogy text file
def load_data(path: Path = Path('./data/google-analogies.txt')) -> List[Raw_Datapoint]:
    dataset = []
    with path.open() as f:
        for line in f.readlines():
            if line[0] == ':':
                task = line[1:].strip(' \n')
                continue

            words = line.strip('\n').split(' ')
            p = Raw_Datapoint(*words, task)
            dataset.append(p)

    return dataset

# Function to transform the raw data into their corresponding word embeddings
def transform_data(analogy_dataset : List[Raw_Datapoint]) -> List[Datapoint]:
    # List to store the transformed datapoints
    transformed_dataset = []
    # For each Raw_Datapoint tuple in the list
    for raw_datapoint in analogy_dataset:
        temp_datapoint = Datapoint()
        # Obtaining a list of the corresponding word embeddings
        embeddings = load_vectors([raw_datapoint.x1, raw_datapoint.x2, raw_datapoint.x3, raw_datapoint.y])
        # Stacking the embeddings horizontally : [1 X 3D]
        temp_datapoint.analogy_embeddings = np.hstack(embeddings[0 : 3])
        # Obtaining the embedding corresponding to y
        temp_datapoint.gt_embedding = embeddings[3]
        # Obtaining the embedding corresponding to z (protected variable)
        temp_datapoint.protected_embedding = obtain_vector_projection(embeddings[3], )
        # Adding to the list of transformed datapoints
        transformed_dataset.append(temp_datapoint)
    # Returning the list of transformed datapoints
    return transformed_dataset
        

