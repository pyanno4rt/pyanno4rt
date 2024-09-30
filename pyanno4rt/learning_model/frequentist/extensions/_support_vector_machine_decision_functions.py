"""Support vector machine decision functions."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import cosh, dot, exp, tanh

# %% Decision functions


def linear_decision_function(svm, features):
    """
    Compute the linear decision function.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the prediction model.

    features : ndarray
        Values of the input features.

    Returns
    -------
    float
        Value of the linear decision function.
    """

    return (dot(features, svm.coef_.T) + svm.intercept_)[0][0]


def rbf_decision_function(svm, features):
    """
    Compute the radial basis decision function.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the prediction model.

    features : ndarray
        Values of the input features.

    Returns
    -------
    float
        Value of the radial basis decision function.
    """

    def kernel(features, support_vectors, gamma):
        """Compute the radial basis kernel."""

        return exp(-gamma*(
            dot(features, features.T)-2*dot(features, support_vectors.T)
            + dot(support_vectors, support_vectors.T)))

    return (sum(svm.dual_coef_[0, i]*kernel(
        features, svm.support_vectors_[i, :], svm.gamma)
        for i, _ in enumerate(svm.support_vectors_)) + svm.intercept_)[0][0]


def poly_decision_function(svm, features):
    """
    Compute the polynomial decision function.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the prediction model.

    features : ndarray
        Values of the input features.

    Returns
    -------
    float
        Value of the polynomial decision function.
    """

    def kernel(features, support_vectors, coef0, degree, gamma):
        """Compute the polynomial kernel."""

        return (gamma*dot(features, support_vectors.T) + coef0)**degree

    return (sum(svm.dual_coef_[0, i]*kernel(
        features, svm.support_vectors_[i, :], svm.coef0, svm.degree, svm.gamma)
        for i, _ in enumerate(svm.support_vectors_)) + svm.intercept_)[0]


def sigmoid_decision_function(svm, features):
    """
    Compute the sigmoid decision function.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the prediction model.

    features : ndarray
        Values of the input features.

    Returns
    -------
    float
        Value of the sigmoid decision function.
    """

    def kernel(features, support_vectors, coef0, gamma):
        """Compute the sigmoid kernel."""

        return tanh(gamma*dot(features, support_vectors.T) + coef0)

    return (sum(svm.dual_coef_[0, i]*kernel(
        features, svm.support_vectors_[i, :], svm.coef0, svm.gamma)
        for i, _ in enumerate(svm.support_vectors_)) + svm.intercept_)[0]

# %% Decision function gradients


def linear_decision_gradient(svm, _):
    """
    Compute the linear decision function gradient.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the prediction model.

    Returns
    -------
    ndarray
        Gradient of the linear decision function.
    """

    return svm.coef_.reshape(-1)


def rbf_decision_gradient(svm, features):
    """
    Compute the radial basis decision function gradient.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the prediction model.

    features : ndarray
        Values of the input features.

    Returns
    -------
    ndarray
        Gradient of the radial basis decision function.
    """

    def kernel_gradient(features, support_vectors, gamma):
        """Compute the rbf kernel gradient."""

        return -2*gamma*(features-support_vectors)*exp(-gamma*(
            dot(features, features.T)-2*dot(features, support_vectors.T)
            + dot(support_vectors, support_vectors.T)))

    return sum(svm.dual_coef_[0, i]*kernel_gradient(
        features, svm.support_vectors_[i, :], svm.gamma)
               for i, _ in enumerate(svm.support_vectors_))[0].reshape(-1)


def poly_decision_gradient(svm, features):
    """
    Compute the polynomial decision function gradient.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the prediction model.

    features : ndarray
        Values of the input features.

    Returns
    -------
    ndarray
        Gradient of the polynomial decision function.
    """

    def kernel_gradient(features, support_vectors, coef0, degree, gamma):
        """Compute the polynomial kernel gradient."""

        return gamma*degree*support_vectors*(
            gamma*dot(features, support_vectors.T) + coef0)**(degree-1)

    return sum(svm.dual_coef_[0, i]*kernel_gradient(
        features, svm.support_vectors_[i, :], svm.coef0, svm.degree, svm.gamma)
               for i, _ in enumerate(svm.support_vectors_)).reshape(-1)


def sigmoid_decision_gradient(svm, features):
    """
    Compute the sigmoid decision function gradient.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the prediction model.

    features : ndarray
        Values of the input features.

    Returns
    -------
    ndarray
        Gradient of the sigmoid decision function.
    """

    def kernel_gradient(features, support_vectors, coef0, gamma):
        """Compute the sigmoid kernel gradient."""

        return gamma*support_vectors*(
            1/cosh(gamma*dot(features, support_vectors.T) + coef0)**2)

    return sum(svm.dual_coef_[0, i]*kernel_gradient(
        features, svm.support_vectors_[i, :], svm.coef0, svm.gamma)
        for i, _ in enumerate(svm.support_vectors_)).reshape(-1)
