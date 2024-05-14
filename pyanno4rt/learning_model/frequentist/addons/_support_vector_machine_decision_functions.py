"""Support vector machine decision functions."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import cosh, dot, exp, power, tanh

# %% Decision functions


def linear_decision_function(svm, features):
    """
    Compute the linear decision function for the SVM.

    Parameters
    ----------
    svm : object of class `SVC`
        Instance of scikit-learn's `SVC` class.

    features : ndarray
        Vector of feature values.

    Returns
    -------
    float
        Value of the decision function with linear kernel.
    """
    return (dot(features, svm.coef_.T) + svm.intercept_)[0][0]


def rbf_decision_function(svm, features):
    """
    Compute the rbf decision function for the SVM.

    Parameters
    ----------
    svm : object of class `SVC`
        Instance of scikit-learn's `SVC` class.

    features : ndarray
        Vector of feature values.

    Returns
    -------
    float
        Value of the decision function with rbf kernel.
    """
    def rbf(features, support_vectors, gamma=1.0/features.shape[1]):
        """Compute the rbf kernel."""
        return exp(-gamma * (
            dot(features, features.T)
            - 2*dot(features, support_vectors.T)
            + dot(support_vectors, support_vectors.T)))

    return (sum(svm.dual_coef_[0, i]
                * rbf(features, svm.support_vectors_[i, :])
                for i, _ in enumerate(svm.support_vectors_))
            + svm.intercept_)[0][0]


def poly_decision_function(svm, features):
    """
    Compute the poly decision function for the SVM.

    Parameters
    ----------
    svm : object of class `SVC`
        Instance of scikit-learn's `SVC` class.

    features : ndarray
        Vector of feature values.

    Returns
    -------
    float
        Value of the decision function with poly kernel.
    """
    def poly(features, support_vectors, coef0, degree,
             gamma=1.0/features.shape[1]):
        """Compute the poly kernel."""
        return power(gamma * dot(features, support_vectors.T) + coef0, degree)

    return (sum(svm.dual_coef_[0, i]
                * poly(features, svm.support_vectors_[i, :],
                       svm.coef0, svm.degree)
                for i, _ in enumerate(svm.support_vectors_))
            + svm.intercept_)[0]


def sigmoid_decision_function(svm, features):
    """
    Compute the sigmoid decision function for the SVM.

    Parameters
    ----------
    svm : object of class `SVC`
        Instance of scikit-learn's `SVC` class.

    features : ndarray
        Vector of feature values.

    Returns
    -------
    float
        Value of the decision function with sigmoid kernel.
    """
    def sigmoid(features, support_vectors, coef0, gamma=1.0/features.shape[1]):
        """Compute the sigmoid kernel."""
        return tanh(gamma * dot(features, support_vectors.T) + coef0)

    return (sum(svm.dual_coef_[0, i]
                * sigmoid(features, svm.support_vectors_[i, :], svm.coef0)
                for i, _ in enumerate(svm.support_vectors_))
            + svm.intercept_)[0]

# %% Decision function gradients


def linear_decision_gradient(svm, _):
    """
    Compute the linear decision function gradient for the SVM.

    Parameters
    ----------
    svm : object of class `SVC`
        Instance of scikit-learn's `SVC` class.

    Returns
    -------
    ndarray
        Gradient of the decision function with linear kernel.
    """
    return svm.coef_.reshape(-1)


def rbf_decision_gradient(svm, features):
    """
    Compute the rbf decision function gradient for the SVM.

    Parameters
    ----------
    svm : object of class `SVC`
        Instance of scikit-learn's `SVC` class.

    features : ndarray
        Vector of feature values.

    Returns
    -------
    ndarray
        Gradient of the decision function with rbf kernel.
    """
    def rbf_gradient(features, support_vectors, gamma=1.0/features.shape[1]):
        """Compute the rbf kernel gradient."""
        return -2 * gamma * (features-support_vectors) * exp(
            -gamma * (dot(features, features.T)
                      - 2*dot(features, support_vectors.T)
                      + dot(support_vectors, support_vectors.T)))

    return sum(svm.dual_coef_[0, i]
               * rbf_gradient(features, svm.support_vectors_[i, :])
               for i, _ in enumerate(svm.support_vectors_))[0].reshape(-1)


def poly_decision_gradient(svm, features):
    """
    Compute the poly decision function gradient for the SVM.

    Parameters
    ----------
    svm : object of class `SVC`
        Instance of scikit-learn's `SVC` class.

    features : ndarray
        Vector of feature values.

    Returns
    -------
    ndarray
        Gradient of the decision function with poly kernel.
    """
    def poly_gradient(features, support_vectors, coef0, degree,
                      gamma=1.0/features.shape[1]):
        """Compute the poly kernel gradient."""
        return gamma * degree * support_vectors * power(
            gamma * dot(features, support_vectors.T) + coef0, degree-1)

    return sum(svm.dual_coef_[0, i]
               * poly_gradient(features, svm.support_vectors_[i, :],
                               svm.coef0, svm.degree)
               for i, _ in enumerate(svm.support_vectors_)).reshape(-1)


def sigmoid_decision_gradient(svm, features):
    """
    Compute the sigmoid decision function gradient for the SVM.

    Parameters
    ----------
    svm : object of class `SVC`
        Instance of scikit-learn's `SVC` class.

    features : ndarray
        Vector of feature values.

    Returns
    -------
    ndarray
        Gradient of the decision function with sigmoid kernel.
    """
    def sigmoid_gradient(features, support_vectors, coef0,
                         gamma=1.0/features.shape[1]):
        """Compute the sigmoid kernel gradient."""
        return gamma * support_vectors * (1 / power(
            cosh(gamma * dot(features, support_vectors.T) + coef0), 2))

    return sum(svm.dual_coef_[0, i]
               * sigmoid_gradient(features, svm.support_vectors_[i, :],
                                  svm.coef0)
               for i, _ in enumerate(svm.support_vectors_)).reshape(-1)
