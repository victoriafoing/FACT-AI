# Importing Libraries
from typing import Dict, List, Tuple
import numpy as np
from sklearn.decomposition import PCA
# from adversarial_debiasing import AdversarialDebiasing
from load_vectors import *
import gensim
import torch
import pickle
from collections import defaultdict

# Function to obtain the male-female gender word pairs
def obtain_gender_pairs(word_vectors : Dict) -> List[List[List]]:
    # List to store the male-female gender word pairs
    pairs = []
    # Defining the word-pairs
    pairs.append(load_vectors(word_vectors, ["he", "she"]))
    pairs.append(load_vectors(word_vectors, ["his", "her"]))
    pairs.append(load_vectors(word_vectors, ["man", "woman"]))
    pairs.append(load_vectors(word_vectors, ["himself", "herself"]))
    pairs.append(load_vectors(word_vectors, ["son", "daughter"]))
    pairs.append(load_vectors(word_vectors, ["father", "mother"]))
    pairs.append(load_vectors(word_vectors, ["guy", "gal"]))
    pairs.append(load_vectors(word_vectors, ["boy", "girl"]))
    pairs.append(load_vectors(word_vectors, ["male", "female"]))
    pairs.append(load_vectors(word_vectors, ["john", "mary"]))
    # Returning the pairs
    return pairs

# Function to obtain the gender subspace or direction
def obtain_gender_subspace(pairs : List[List[List]], components: int = 1, use_boluk : bool = False) -> np.ndarray:
    # Obtaining an object of the PCA class
    pca = PCA(n_components = components)
    # If we want to use the bolukbasi method
    if use_boluk:
        # Obtaining the means of each pair
        # List of numpy arrays (means)
        means = [np.sum(np.array(x), axis = 0) / np.array(x).shape[0] for x in pairs]
        # Obtaining the differences
        # Numpy array 20 X Word_embedding_dim
        differences = np.array([np.array(x) - means[i] for i in range(len(means)) for x in pairs[i]])
    else:
        # Obtaining the difference vectors of each pair
        differences = np.array([np.array(x[0]) - np.array(x[1]) for x in pairs])
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

# Function to obtain the 10 most similar words to the predicted embedding
def obtain_most_similar(a : np.ndarray, word_vectors : Dict, similar_count : int = 10) -> List[List[tuple]]:
    # Resolving memory issues
    word_vectors.init_sims(replace = True)
    # If only a single predicted embedding has been passed
    if len(a.shape) == 1:
        a = np.reshape(a, (1, -1))
    # For each predicted embedding
    most_similar_list = [word_vectors.similar_by_vector(a[i], topn = similar_count) for i in range(a.shape[0])]
    # Returning the list
    return most_similar_list

# Function to convert the Google news bin format into the word2vec format
def load_word2vec_format(f, max_num_words=None):
  """Loads word2vec data from a file handle.

  Similar to gensim.models.keyedvectors.KeyedVectors.load_word2vec_format
  but takes a file handle as input rather than a filename. This lets us use
  GFile. Also only accepts binary files.

  Args:
    f: file handle
    max_num_words: number of words to load. If None, load all.

  Returns:
    Word2vec data as keyedvectors.EuclideanKeyedVectors.
  """
  header = f.readline()
  vocab_size, vector_size = (
      int(x) for x in header.rstrip().split())  # throws for invalid file format
  print("vector_size",vector_size)
  result = gensim.models.keyedvectors.EuclideanKeyedVectors()
  num_words = 0
  result.vector_size = vector_size
  result.syn0 = np.zeros((vocab_size, vector_size), dtype=np.float32)

  def add_word(word, weights):
    word_id = len(result.vocab)
    if word in result.vocab:
      print("duplicate word '%s', ignoring all but first", word)
      return
    result.vocab[word] = gensim.models.keyedvectors.Vocab(
        index=word_id, count=vocab_size - word_id)
    result.syn0[word_id] = weights
    result.index2word.append(word)

  if max_num_words and max_num_words < vocab_size:
    num_embeddings = max_num_words
  else:
    num_embeddings = vocab_size
  print("Loading ",num_embeddings," embeddings")

  binary_len = np.dtype(np.float32).itemsize * vector_size
  for _ in range(vocab_size):
    # mixed text and binary: read text first, then binary
    word = []
    while True:
      ch = f.read(1)
      if ch == b' ':
        break
      if ch == b'':
        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
      if ch != b'\n':  # ignore newlines in front of words (some binary files have)
        word.append(ch)
    word = gensim.utils.to_unicode(b''.join(word), encoding='utf-8', errors='strict')
    weights = np.frombuffer(f.read(binary_len), dtype=np.float32)
    add_word(word, weights)
    num_words = num_words + 1
    if max_num_words and num_words == max_num_words:
      break
  if result.syn0.shape[0] != len(result.vocab):
    print(
        "duplicate words detected, shrinking matrix size from %i to %i",
        result.syn0.shape[0], len(result.vocab))
  result.syn0 = np.ascontiguousarray(result.syn0[:len(result.vocab)])
  assert (len(result.vocab), vector_size) == result.syn0.shape

  print("loaded %s matrix", result.syn0.shape)
  return result

# Function to reset the seeds
def reset_seeds(seed : int = 42) -> None:
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

# Function to perform grid search
def grid_search(learning_rate_list : List[float], adversary_loss_weight_list : List[float], \
  word_embedding_dim : int, gender_subspace : np.ndarray, transformed_analogy_dataset, word_embedding_type : str, file_path : str) -> None:

  # For each learning rate in the list
  for learning_rate in learning_rate_list:

    # For each adversary loss weight (alpha) in the list
    for adversary_loss_weight in adversary_loss_weight_list:

      # Reset seeds
      reset_seeds()

      # Creating an instance of the non-debiased model with the intended configuration
      non_debiased_model = AdversarialDebiasing(
        seed = 42,
        word_embedding_dim = word_embedding_dim,
        num_epochs = 500,
        debias = False,
        gender_subspace = gender_subspace,
        batch_size = 256,
        adversary_loss_weight = adversary_loss_weight,
        classifier_learning_rate = learning_rate,
        adversary_learning_rate = learning_rate
      )

      # Fitting to the dataset and training the non-debiased model
      non_debiased_model.fit(dataset = transformed_analogy_dataset)

      # Dictionary for storing the best weights of the non-debiased model
      best_dict = {"W1" : non_debiased_model.best_W1, "W2" : non_debiased_model.best_W2}
      
      # Saving the dictionary
      with open(os.path.join(file_path, 'non_debiased', "_".join([word_embedding_type, str(learning_rate), str(adversary_loss_weight), 'best']) + '.pckl'), 'wb') as f:
        pickle.dump(best_dict, f)

      # Dictionary for saving the final weights of the non-debiased model  
      last_dict = {"W1" : non_debiased_model.W1, "W2" : non_debiased_model.W2}

      # Saving the dictionary
      with open(os.path.join(file_path, 'non_debiased', "_".join([word_embedding_type, str(learning_rate), str(adversary_loss_weight), 'last']) + '.pckl'), 'wb') as f:
        pickle.dump(last_dict, f)

      # Reset seeds
      reset_seeds()
        
      # Creating an instance of the debiased model with the intended configuration
      debiased_model = AdversarialDebiasing(
          seed = 42,
          word_embedding_dim = word_embedding_dim,
          num_epochs = 500,
          debias = True,
          gender_subspace = gender_subspace,
          batch_size = 256,
          adversary_loss_weight = adversary_loss_weight,
          classifier_learning_rate = learning_rate,
          adversary_learning_rate = learning_rate
      )

      # Fitting to the dataset and training the debiased model
      debiased_model.fit(dataset = transformed_analogy_dataset)

      # Dictionary for storing the best weights of the debiased model
      best_dict = {"W1" : debiased_model.best_W1, "W2" : debiased_model.best_W2}
      
      # Saving the dictionary
      with open(os.path.join(file_path, 'debiased', "_".join([word_embedding_type, str(learning_rate), str(adversary_loss_weight), 'best']) + '.pckl'), 'wb') as f:
        pickle.dump(best_dict, f)

      # Dictionary for saving the final weights of the debiased model  
      last_dict = {"W1" : debiased_model.W1, "W2" : debiased_model.W2}

      # Saving the dictionary
      with open(os.path.join(file_path, 'debiased', "_".join([word_embedding_type, str(learning_rate), str(adversary_loss_weight), 'last']) + '.pckl'), 'wb') as f:
        pickle.dump(last_dict, f)
  
# Function to obtain the trained parameters
def obtain_trained_parameters(file_path : str) -> Dict:

  # Initializing the dictionary
  pretrained_parameters = defaultdict(lambda : defaultdict(dict))

  # Obtaining the parameters for each model type
  for model_type in ["non_debiased", "debiased"]:

    # For each saved dictionary in the respective folder
    for saved_dict_file in os.listdir(os.path.join(file_path, model_type)):

      # Obtaining each component of the file name
      saved_dict_file_split = saved_dict_file.split('_')

      # Loading the saved dictionary
      temp_dict = torch.load(os.path.join(file_path, model_type, saved_dict_file))
      
      # Setting the corresponding element
      pretrained_parameters[model_type][saved_dict_file_split[0]]["W1"] = temp_dict["W1"]
  
  # Returning the dictionary of saved parameters
  return pretrained_parameters
    

    