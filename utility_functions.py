# Importing Libraries
from typing import Dict, List, Tuple
import numpy as np
from sklearn.decomposition import PCA
from load_vectors import *

# Function to obtain the male-female gender word pairs
def obtain_gender_pairs(data_path : str, save_dir : str, save_file : str) -> List[List]:
    # List to store the male-female gender word pairs
    pairs = []
    # Defining the word-pairs
    pairs.append(load_vectors(data_path, save_dir, save_file, ["he", "she"]))
    pairs.append(load_vectors(data_path, save_dir, save_file, ["his", "her"]))
    pairs.append(load_vectors(data_path, save_dir, save_file, ["man", "woman"]))
    pairs.append(load_vectors(data_path, save_dir, save_file, ["himself", "herself"]))
    pairs.append(load_vectors(data_path, save_dir, save_file, ["son", "daughter"]))
    pairs.append(load_vectors(data_path, save_dir, save_file, ["father", "mother"]))
    pairs.append(load_vectors(data_path, save_dir, save_file, ["guy", "gal"]))
    pairs.append(load_vectors(data_path, save_dir, save_file, ["boy", "girl"]))
    pairs.append(load_vectors(data_path, save_dir, save_file, ["male", "female"]))
    pairs.append(load_vectors(data_path, save_dir, save_file, ["John", "Mary"]))
    # Returning the pairs
    return pairs

# Function to obtain the gender subspace or direction
def obtain_gender_subspace(pairs : List[List], components: int = 1) -> np.ndarray:
    # Obtaining the means of each pair
    means = [sum(x) / len(x) for x in pairs]
    # Obtaining the differences
    differences = np.array([x - means[i] for i in range(len(means)) for x in pairs[i]])
    # Obtaining an object of the PCA class
    pca = PCA(n_components = components)
    # Fitting the PCA object onto the difference matrix
    pca.fit(differences)
    # Returning the PCA components spanning the gender / bias subspace
    return pca.components_

# Function to project a vector A in the direction of a vector B or a list of vectors B
def obtain_vector_projection(a : np.ndarray, b : np.ndarray) -> np.ndarray:
    # Component of vector a along each spanning component
    a_components = a.dot(b.T)
    # Projections of vector along each spanning component
    # Each column represents the projection vector along a spanning component
    a_projections = np.multiply(a_components, b.T)
    # Each row represents the projection vector along a spanning component
    a_projections = a_projections.T
    # Summing along the rows
    a_projections = np.sum(a_projections, axis = 0)
    # Returning the overall projection vector
    return a_projections