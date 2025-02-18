import math
from typing import List

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import Optimizer, Adam
import pickle

# Reset seeds
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
                 adversary_loss_weight=1.0,
                 num_epochs=500,
                 batch_size=1000,
                 debias=True,
                 word_embedding_dim=100,
                 classifier_learning_rate = 2 ** -16,
                 adversary_learning_rate = 2 ** -16,
                 gender_subspace = None):
        """
        Args:
            seed (int, optional): Seed to make `predict` repeatable.
            adversary_loss_weight (float, optional): Hyperparameter that chooses
                the strength of the adversarial loss.
            num_epochs (int, optional): Number of training epochs.
            batch_size (int, optional): Batch size.
            debias (bool, optional): Learn a classifier with or without
                debiasing.
            word_embedding_dim: Dimensionality of the word embeddings
            classifier_learning_rate: Learning rate for the predictor model
            adversary_learning_rate: Learning rate for the adversary model
            gender_subspace: Unit vector(s) spanning the gender direction
        """
        self.seed = seed

        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = int(batch_size)
        self.debias = debias
        self.word_embedding_dim = word_embedding_dim
        self.classifier_learning_rate = classifier_learning_rate
        self.adversary_learning_rate = adversary_learning_rate
        self.gender_subspace = gender_subspace

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.W1 = torch.Tensor(self.word_embedding_dim, 1).to(device=self.device)
        self.W1 = nn.Parameter(nn.init.xavier_uniform_(self.W1))

        self.W2 = torch.Tensor(self.word_embedding_dim, 1).to(device=self.device)
        self.W2 = nn.Parameter(nn.init.normal_(self.W2, mean=0.000001, std=0.00001))

        self.best_score = 0
        self.best_W1 = self.W1.clone()
        self.best_W2 = self.W2.clone()

        self.classifier_vars = [self.W1]
        self.adversary_vars = [self.W2]

        self.losses = {
            'predictor': [],
            'adversary': [],
        }

    def _classifier_model(self, features):
        """Compute the classifier predictions for the outcome variable.
        """
        x1 = features[:, 0:self.word_embedding_dim]
        x2 = features[:, self.word_embedding_dim:self.word_embedding_dim * 2]
        x3 = features[:, self.word_embedding_dim * 2:self.word_embedding_dim * 3]

        v = x2 + x3 - x1
        predicted_word = v - F.linear(v, self.W1 @ self.W1.transpose(0, 1))

        return predicted_word

    def _adversary_model(self, pred_logits):
        """Compute the adversary predictions for the protected attribute.
        """
        pred_protected = F.linear(pred_logits, self.W2.transpose(0, 1))

        return pred_protected

    def fit(self, dataset: List[Datapoint]):
        """Compute the model parameters of the fair classifier using gradient
        descent.
        Args:
            dataset: Dataset containing true labels.
        Returns:
            AdversarialDebiasing: Returns self.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Obtain classifier predictions and classifier loss
        # starter_learning_rate = 0.001
        classifier_optimizer = optim.Adam([self.W1], lr=self.classifier_learning_rate)
        adversary_optimizer = optim.Adam([self.W2], lr=self.adversary_learning_rate)

        predictor_lr_scheduler = optim.lr_scheduler.ExponentialLR(classifier_optimizer, 0.96)
        if self.debias:
            adversary_lr_scheduler = optim.lr_scheduler.ExponentialLR(adversary_optimizer, 0.96)

        num_train_samples = len(dataset)

        for epoch in range(self.num_epochs):
            print(f"[{epoch}/{self.num_epochs}] Running epoch")

            # All shuffled ids
            shuffled_ids = np.random.choice(num_train_samples, num_train_samples).astype(int)

            self._train_epoch(dataset, shuffled_ids, classifier_optimizer, adversary_optimizer, num_train_samples, epoch)

            predictor_lr_scheduler.step()
            if self.debias:
                adversary_lr_scheduler.step()

            if epoch % 10 == 0:
                print(f"||w||: {np.linalg.norm(self.W1.cpu().clone().detach())}")
                print(f"||w2||: {np.linalg.norm(self.W2.cpu().clone().detach())}")
                print(f"w.T g: {np.dot(self.W1.clone().detach().cpu().numpy().T, self.gender_subspace.T)}")

        return self

    def _train_epoch(self, dataset: List[Datapoint], shuffled_ids: np.ndarray, classifier_optimizer: Adam, adversary_optimizer: Adam, num_train_samples: int,
                     epoch: int):
        """ Train the model for one epoch

        Args:
            dataset: The training dataset containing true labels
            shuffled_ids: Shuffled IDs of the training samples
            classifier_optimizer: Type of optimizer for the predictor model
            adversary_optimizer: Type of optimizer for the adversary model
            num_train_samples: Number of training samples
            epoch: The epoch number
        """
        for i in range(math.floor(num_train_samples // self.batch_size)):
            batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)].astype(int)

            data_points: List[Datapoint] = [dataset[i] for i in batch_ids]

            # batch_features: N x (WordEmbeddingDim * 3)
            batch_features = torch.cat([torch.Tensor(x.analogy_embeddings).unsqueeze_(0) for x in data_points]).to(device=self.device)

            # batch_labels: N x VocabularyDim
            batch_labels = torch.cat([torch.Tensor(x.gt_embedding).unsqueeze_(0) for x in data_points]).to(device=self.device)

            # batch_protected_attributes: N x 1 (?)
            batch_protected_labels = torch.cat([torch.Tensor(x.protected) for x in data_points]).to(device=self.device)


            # Run the classifier
            pred_embeddings = self._classifier_model(batch_features)
            pred_labels_loss = F.mse_loss(pred_embeddings, batch_labels)

            # Run the adversary
            if self.debias:
                pred_protected = self._adversary_model(pred_embeddings)
                pred_protected = pred_protected.squeeze()
                pred_protected_loss = F.mse_loss(pred_protected, batch_protected_labels)

            # Zero the gradients
            if self.debias:
                adversary_optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            # Calculate the gradients for the adversary
            if self.debias:
                # Calculate adversary gradients with respect to W1?
                pred_protected_loss.backward(retain_graph=True)
                adversary_grads = {var: var.grad.clone() for var in self.classifier_vars}

            # Optimize the Classifier with the gradients of the classifier and adversary
            classifier_optimizer.zero_grad()
            pred_labels_loss.backward()
            # for classifier_var in self.classifier_vars:
            if self.debias:
                unit_adversary_grad = normalize(adversary_grads[self.W1])
                self.W1.grad -= torch.sum(self.W1.grad * unit_adversary_grad) * unit_adversary_grad
                self.W1.grad -= self.adversary_loss_weight * adversary_grads[self.W1]
            classifier_optimizer.step()

            # Update adversary parameters by the gradient of pred_protected_attributes_loss
            if self.debias:
                adversary_optimizer.step()
                pass

            self.losses['predictor'].append(pred_labels_loss.item())

            if self.debias:
                self.losses['adversary'].append(pred_protected_loss.item())

            val_metric = abs(np.dot(self.W1.clone().detach().cpu().numpy().T, self.gender_subspace.T))

            if val_metric > self.best_score:
                self.best_score = val_metric
                self.best_W1 = self.W1.clone()
                self.best_W2 = self.W2.clone()
            
            if self.debias and i % 10 == 0:
                print("epoch %d; iter: %d; batch classifier loss: %f; batch adversarial loss: %f" % (
                    epoch, i, pred_labels_loss.item(), pred_protected_loss.item()))
            elif i % 200 == 0:
                print("epoch %d; iter: %d; batch classifier loss: %f" % (
                    epoch, i, pred_labels_loss.item()))

    def predict(self,  datapoints: np.ndarray) -> np.ndarray:
        """Obtain the predictions for the provided dataset using the fair
        classifier learned.
        Args:
            dataset: Dataset containing incomplete analogies that need
                to be predicted.
        Returns:
            predictions: Predictions pertaining to the incomplete analogies.
        """
        batch_features = torch.cat([torch.Tensor(x).unsqueeze_(0) for x in datapoints]).to(device=self.device)
        predictions = self._classifier_model(batch_features)

        return predictions.detach().cpu().numpy()

    def get_model_weights(self):
        return self.W1

def normalize(x):
    return x / (torch.norm(x) + np.finfo(np.float32).tiny)
