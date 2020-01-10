# Importing Libraries
from typing import Dict
import numpy as np
from sklearn.decomposition import PCA

# Function to obtain the gender subspace or direction
def obtain_gender_subspace(pairs : List, embeddings : Dict, components = 1 : int) -> np.ndarray:
    # # Defining the 10 pairs of (male, female) words
    # sets = [(embeddings['he'], embeddings['she']), \
    #     (embeddings['his'], embeddings['her']), \
    #     (embeddings['man'], embeddings['woman']), \
    #     (embeddings['himself'], embeddings['herself']), \
    #     (embeddings['son'], embeddings['daughter']), \
    #     (embeddings['father'], embeddings['mother']), \
    #     (embeddings['guy'], embeddings['gal']), \
    #     (embeddings['boy'], embeddings['girl']), \
    #     (embeddings['male'], embeddings['female']), \
    #     (embeddings['John'], embeddings['Mary'])]
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