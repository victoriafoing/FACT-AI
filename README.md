# Adversarial Debiasing
This repository contains the PyTorch implementation of the **Word Embeddings** experiment as proposed in the paper *"Mitigating Unwanted Biases with Adversarial Learning"*, B. H. Zhang, B. Lemoine, and M. Mitchell, AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society, 2018 [[link](https://arxiv.org/abs/1801.07593)].

The paper proposes an adversarial learning technique to mitigate the effect of biases generally present in the training data and the **Word Embeddings** experiment focuses on removing gender bias from word embeddings.

Further extensions have been made to the above experiment by testing the proposed technique on word embeddings trained on different domain data namely, *Wikipedia2Vec* and *Glove* in addition to *Word2Vec / GoogleNews* (that was originally used in the paper) along with qualitative and quantitative evaluations. For example, the effect of applying the debiasing technique on our model trained on GloVe embeddings can be seen below for the test analogy 

**he : director :: she : ?**

|Biased Neighbour| Biased Similarity|Debiased Neighbour | Debiased Similarity|
| -------------- | ---------------- | ----------------- | ------------------ |
|    assistant   |       0.642      |    co-director    |      0.571         |
|     julie      |       0.635      |    co-artistic    |      0.483         | 
|    executive   |       0.629      | president/creative|      0.474         |
|   coordinator  |       0.625      |   co-ordinator    |      0.468         |
|    associate   |       0.623      |    chairwoman     |      0.463         |
|     susan      |       0.616      |      debra        |      0.458         |
|   directors    |       0.608      |    coordinator    |      0.453         |
|    jennifer    |       0.605      |   choreographer   |      0.452         |
|    directed    |       0.599      |     kathryn       |      0.450         |

## Authors

Rochelle Choenni (10999949, rmvk97@gmail.com)
<br />
Maximilian Filtenborg (11042729, max.filtenborg@gmail.com)
<br />
Victoria Foing (11773391, vickyfoing@gmail.com)
<br />
Gaurav Kudva (12205583, gauravkudva2@gmail.com)

## Teaching Assistant

Leon Lang (leon.lang@student.uva.nl)

## Environment Setup

We have a conda `environment.yml` file specifying the required packages:

    $ conda env create -f environment.yml

And activate the environment:

    $ conda activate fact
   
Update your environment later (after activating it):

    $ conda env update --file environment.yml

Update your env with the requirements.txt (packages only in pip):

    $ pip install -r requirements.txt

Update the env after installing a package:

    $ conda env export --from-history > environment.yml

If you add a package from pip, add it to the `requirements.txt` file.

## Files
`data` is a folder containing the Google Analogies Dataset and the qualitative evaluation examples. This is also the folder where the word embeddings are downloaded and saved to, in case they do not already exist in the same.

`models` is a folder containing the pre-trained weights pertaining to the best hyperparameter configuration per word embedding per model type (biased / debiased).

`adversarial_debiasing.py` contains the class and function definitions of the predictor and adversary models.

`demo_adversarial_debiasing.ipynb` is a jupyter notebook showing a demonstration of the qualitative evaluation on the biased and debiased models. You can either choose to use a pre-trained model or train a model from scratch on either of the 3 word embeddings proposed above. **This is the main notebook for the purpose of demonstration.**

`environment.yml` is the environment configuration file.

`experiments_adversarial_debiasing.ipynb` is a jupyter notebook that contains the grid search experimentation pertaining to the best hyperparameter configuration per word embedding per model type (biased / debiased).

`load_data.py` contains the functions used to load, pre-process and transform the Google Analogies Dataset for our experimentation.

`load_vectors.py` contains the functions used to download, pre-process and load the word embeddings in consideration.

`qualitative_evaluation.py` contains the functions used to display the results of the qualitative evalution.

`requirements.txt` is the enviroment configuration file (pertaining to packages installed using `pip` instead of `conda`).

`utility_functions.py` contains the main helper functions used throughout the experimentation phase.

## Practical Considerations
> Our code downloads the desired word embeddings file in the event that it is not already present in the `data` folder.

> We found that the downloading and pre-processing of `Wikipedia2Vec` took a substantial amount of time during our experimentation phase. We advise you to kindly consider using either `Glove` or `Word2Vec / GoogleNews` embeddings in our demonstration notebook, in case you prefer to see the results relatively quickly.
