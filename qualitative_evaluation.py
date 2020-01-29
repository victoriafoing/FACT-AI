# Importing Libraries
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch
import os

# Examples to test the models upon
def get_datapoints(word_vectors):
    datapoints, test_analogies = [], []
    with open(os.path.join('data', 'sexism-traps.txt'), 'r') as f:
        # Reading each line
        for line in f.readlines():
            words = line.split()
            if words[0] == ':':
                continue
            test_analogies.append(words)
            word_embeddings = word_vectors[words]
            word_embeddings = np.reshape(word_embeddings, (1, -1))
            datapoints.append(word_embeddings)
    datapoints = np.vstack(datapoints)
    return datapoints, test_analogies

# Qualitative evaluation of the non-debiased model
def get_non_debiased_predictions(datapoints, word_embedding_dim):
    features = torch.cat([torch.Tensor(x).unsqueeze_(0) for x in datapoints])
    x1 = features[:, 0:word_embedding_dim]
    x2 = features[:, word_embedding_dim:word_embedding_dim * 2]
    x3 = features[:, word_embedding_dim * 2:word_embedding_dim * 3]

    non_debiased_predictions = x2 + x3 - x1
    non_debiased_predictions = non_debiased_predictions.cpu().detach().numpy()
    return non_debiased_predictions

# Displaying the similarity list for the non-debiased model
def print_singular_table(most_similar_list, test_analogies):
    most_similar_list_data_frames = []
    for i in range(len(most_similar_list)):
        print("{} : {} :: {} : ".format(test_analogies[i][0], test_analogies[i][1], test_analogies[i][2]))
        temp_data_frame = pd.DataFrame(most_similar_list[i][1:], columns = ['Neighbor', 'Similarity'])
        most_similar_list_data_frames.append(temp_data_frame)
        print(tabulate(temp_data_frame, headers='keys', tablefmt='psql', showindex=False))

# Combining the dataframes pertaining to both the variants of the model
def print_combined_table(non_debiased_most_similar_list, debiased_most_similar_list, test_analogies):
    iterables = [['Biased', 'Debiased'], ['Neighbour', 'Similarity']]
    index = pd.MultiIndex.from_product(iterables)
    overall_data_frames_list = []
    for i in range(len(non_debiased_most_similar_list)):
        overall_list = []
        print("{} : {} :: {} : ".format(test_analogies[i][0], test_analogies[i][1], test_analogies[i][2]))
        for j in range(1, len(non_debiased_most_similar_list[i])):
            temp_list = []
            temp_list.append(non_debiased_most_similar_list[i][j][0])
            temp_list.append(round(non_debiased_most_similar_list[i][j][1], 3))
            temp_list.append(debiased_most_similar_list[i][j][0])
            temp_list.append(round(debiased_most_similar_list[i][j][1], 3))
            overall_list.append(temp_list)
        temp_df = pd.DataFrame(overall_list, columns = index)
        print(tabulate(temp_df, headers = ['Biased\nNeighbour', 'Biased\nSimilarity', 'Debiased\nNeighbour', 'Debiased\nSimilarity'], tablefmt = 'psql', showindex = False))
        overall_data_frames_list.append(temp_df)