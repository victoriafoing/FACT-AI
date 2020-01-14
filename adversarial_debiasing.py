import math
from typing import List

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import Optimizer, Adam

from load_data import Datapoint


class AdversarialDebiasing:
    """Adversarial debiasing is an in-processing technique that learns a
    classifier to maximize prediction accuracy and simultaneously reduce an
    adversary's ability to determine the protected attribute from the
    predictions [5]_. This approach leads to a fair classifier as the
    predictions cannot carry any group discrimination information that the
    adversary can exploit.
    References:
        .. [5] B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating Unwanted
           Biases with Adversarial Learning," AAAI/ACM Conference on Artificial
           Intelligence, Ethics, and Society, 2018.
    """

    def __init__(self,
                 seed=None,
                 adversary_loss_weight=0.1,
                 num_epochs=50,
                 batch_size=128,
                 classifier_num_hidden_units=200,
                 debias=True,
                 word_embedding_dim=100):
        """
        Args:
            unprivileged_groups (tuple): Representation for unprivileged groups
            privileged_groups (tuple): Representation for privileged groups
            seed (int, optional): Seed to make `predict` repeatable.
            adversary_loss_weight (float, optional): Hyperparameter that chooses
                the strength of the adversarial loss.
            num_epochs (int, optional): Number of training epochs.
            batch_size (int, optional): Batch size.
            classifier_num_hidden_units (int, optional): Number of hidden units
                in the classifier model.
            debias (bool, optional): Learn a classifier with or without
                debiasing.
        """
        # super(AdversarialDebiasing, self).__init__(
        #     unprivileged_groups=unprivileged_groups,
        #     privileged_groups=privileged_groups)
        self.seed = seed

        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = int(batch_size)
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias
        self.word_embedding_dim = word_embedding_dim

        # self.features_ph = None
        # self.protected_attributes_ph = None
        # self.true_labels_ph = None
        # self.pred_labels = None

        self.W1 = torch.Tensor(self.word_embedding_dim, 1)
        self.W1 = nn.Parameter(nn.init.xavier_uniform_(self.W1))

        self.W2 = torch.Tensor(self.word_embedding_dim, 1)
        self.W2 = nn.Parameter(nn.init.xavier_uniform_(self.W2))

        self.classifier_vars = [self.W1]
        self.adversary_vars = [self.W2]

    def _classifier_model(self, features):
        """Compute the classifier predictions for the outcome variable.
        """
        x1 = features[:, 0:self.word_embedding_dim]
        x2 = features[:, self.word_embedding_dim:self.word_embedding_dim * 2]
        x3 = features[:, self.word_embedding_dim * 2:self.word_embedding_dim * 3]

        v = x1 + x2 - x3
        predicted_word = v - F.linear(v, self.W1 @ self.W1.transpose(0, 1))

        return predicted_word

    def _adversary_model(self, pred_logits):
        """Compute the adversary predictions for the protected attribute.
        """
        pred_protected_embedding = F.linear(pred_logits, self.W2.transpose(0, 1))

        return pred_protected_embedding

    def fit(self, dataset: List[Datapoint]):
        """Compute the model parameters of the fair classifier using gradient
        descent.
        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.
        Returns:
            AdversarialDebiasing: Returns self.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Obtain classifier predictions and classifier loss
        starter_learning_rate = 0.001
        classifier_optimizer = optim.Adam([self.W1], lr=starter_learning_rate)
        adversary_optimizer = optim.Adam([self.W2], lr=starter_learning_rate)

        predictor_lr_scheduler = optim.lr_scheduler.ExponentialLR(classifier_optimizer, 0.96)
        adversary_lr_scheduler = optim.lr_scheduler.ExponentialLR(adversary_optimizer, 0.96)

        num_train_samples = len(dataset)

        for epoch in range(self.num_epochs):
            print(f"[{epoch}/{self.num_epochs}] Running epoch")

            # All shuffled ids
            shuffled_ids = np.random.choice(num_train_samples, num_train_samples).astype(int)

            self.train(dataset, shuffled_ids, classifier_optimizer, adversary_optimizer, num_train_samples, epoch)

            predictor_lr_scheduler.step()
            adversary_lr_scheduler.step()

        return self

    def train(self, dataset: List[Datapoint], shuffled_ids: np.ndarray, classifier_optimizer: Adam, adversary_optimizer: Adam, num_train_samples: int,
              epoch: int):
        """ Train the model for one epoch

        Args:
            dataset:
            shuffled_ids:
            optimizer:
            num_train_samples:
            epoch:
        """
        print(self.batch_size, num_train_samples, range(num_train_samples // self.batch_size))
        for i in range(math.ceil(num_train_samples // self.batch_size)):
            batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)].astype(int)

            data_points: List[Datapoint] = [dataset[i] for i in batch_ids]

            # Batch of features
            # batch_features: N x (WordEmbeddingDim * 3)
            batch_features = torch.cat([torch.Tensor(x.analogy_embeddings).unsqueeze_(0) for x in data_points])

            # One-hot batch represetnation of a batch of labels
            # batch_labels: N x VocabularyDim
            # TODO: Batch labels should actually be the one-hot vector of labels of size vocabulary, not gt_embedding
            batch_labels = torch.cat([torch.Tensor(x.gt_embedding).unsqueeze_(0) for x in data_points])

            # Batch of protected attributes
            # batch_protected_attributes: N x 1 (?)
            batch_protected_embeddings = torch.cat([torch.Tensor(x.protected_embedding).unsqueeze_(0) for x in data_points])

            # Run the classifier
            pred_embeddings = self._classifier_model(batch_features)
            pred_labels_loss = F.mse_loss(pred_embeddings, batch_labels)

            # Run the adversary
            pred_protected_embeddings = self._adversary_model(pred_embeddings)
            pred_protected_embeddings_loss = F.mse_loss(pred_protected_embeddings, batch_protected_embeddings)

            # Zero the gradients
            adversary_optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            # Calculate the gradients for the adversary
            pred_protected_embeddings_loss.backward(retain_graph=True)
            adversary_grads = {var: var.grad for var in self.classifier_vars}

            # Optimize the Classifier with the gradients of the classifier and adversary
            pred_labels_loss.backward()
            for classifier_var in self.classifier_vars:
                grad = classifier_var.grad
                if self.debias:
                    unit_adversary_grad = normalize(adversary_grads[classifier_var])
                    grad -= torch.sum(grad * unit_adversary_grad) * unit_adversary_grad
                    grad -= self.adversary_loss_weight * adversary_grads[classifier_var]
            classifier_optimizer.step()

            # Update adversary parameters by the gradient of pred_protected_attributes_loss
            if self.debias:
                adversary_optimizer.step()

            if self.debias and i % 10 == 0:
                print("epoch %d; iter: %d; batch classifier loss: %f; batch adversarial loss: %f" % (
                    epoch, i, pred_labels_loss.item(), pred_protected_embeddings_loss.item()))
            elif i % 200 == 0:
                print("epoch %d; iter: %d; batch classifier loss: %f" % (
                    epoch, i, pred_labels_loss.item()))

    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the fair
        classifier learned.
        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs
                to be transformed.
        Returns:
            dataset (BinaryLabelDataset): Transformed dataset.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        num_test_samples, _ = np.shape(dataset.features)

        samples_covered = 0
        pred_labels = []
        while samples_covered < num_test_samples:
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > num_test_samples:
                end = num_test_samples
            batch_ids = np.arange(start, end)
            batch_features = dataset.features[batch_ids]
            batch_labels = np.reshape(dataset.labels[batch_ids], [-1, 1])
            batch_protected_attributes = np.reshape(dataset.protected_attributes[batch_ids][:,
                                                    dataset.protected_attribute_names.index(self.protected_attribute_name)], [-1, 1])

            batch_feed_dict = {self.features_ph: batch_features,
                               self.true_labels_ph: batch_labels,
                               self.protected_attributes_ph: batch_protected_attributes,
                               self.keep_prob: 1.0}

            pred_labels += self.sess.run(self.pred_labels, feed_dict=batch_feed_dict)[:, 0].tolist()
            samples_covered += len(batch_features)

        # Mutated, fairer dataset with new labels
        dataset_new = dataset.copy(deepcopy=True)
        dataset_new.labels = (np.array(pred_labels) > 0.5).astype(np.float64).reshape(-1, 1)

        # Map the dataset labels to back to their original values.
        temp_labels = dataset_new.labels.copy()

        temp_labels[(dataset_new.labels == 1.0).ravel(), 0] = dataset.favorable_label
        temp_labels[(dataset_new.labels == 0.0).ravel(), 0] = dataset.unfavorable_label

        dataset_new.labels = temp_labels.copy()

        return dataset_new


def normalize(x):
    return x / (torch.norm(x) + np.finfo(np.float32).tiny)
