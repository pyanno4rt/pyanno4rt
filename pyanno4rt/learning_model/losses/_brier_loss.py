"""Brier score loss function."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from sklearn.metrics import brier_score_loss

# %% Function definition


def brier_loss(true_labels, predicted_labels):
    """Compute the log loss from the true and predicted labels."""
    return -brier_score_loss(true_labels, predicted_labels)
