"""Support vector machine model kernel-based decision function gradients."""

# Author: Tim Ortkamp

# %% External package import

from numpy import cosh, dot, exp

# %% Kernel-based gradients


def linear_gradient(svm, _):
    """
    Compute the linear kernel SVM decision function gradient.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the prediction model.

    Returns
    -------
    ndarray
        Linear kernel SVM decision function gradient.
    """

    return svm.coef_.reshape(-1)


def rbf_gradient(svm, features):
    """
    Compute the RBF kernel SVM decision function gradient.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the prediction model.

    features : ndarray
        Feature values.

    Returns
    -------
    ndarray
        RBF kernel SVM decision function gradient.
    """

    def kernel_gradient(features, support_vectors, gamma):
        """Compute the RBF kernel gradient."""

        return -2*gamma*(features-support_vectors)*exp(-gamma*(
            dot(features, features.T)-2*dot(features, support_vectors.T)
            + dot(support_vectors, support_vectors.T)))

    return sum(
        svm.dual_coef_[0, i]*kernel_gradient(
            features, svm.support_vectors_[i, :], svm.gamma)
        for i, _ in enumerate(svm.support_vectors_))[0].reshape(-1)


def poly_gradient(svm, features):
    """
    Compute the polynomial kernel SVM decision function gradient.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the prediction model.

    features : ndarray
        Feature values.

    Returns
    -------
    ndarray
        Polynomial kernel SVM decision function gradient.
    """

    def kernel_gradient(features, support_vectors, coef0, degree, gamma):
        """Compute the polynomial kernel gradient."""

        return gamma*degree*support_vectors*(
            gamma*dot(features, support_vectors.T) + coef0)**(degree-1)

    return sum(
        svm.dual_coef_[0, i]*kernel_gradient(
            features, svm.support_vectors_[i, :], svm.coef0, svm.degree,
            svm.gamma)
        for i, _ in enumerate(svm.support_vectors_)).reshape(-1)


def sigmoid_gradient(svm, features):
    """
    Compute the sigmoid kernel SVM decision function gradient.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the prediction model.

    features : ndarray
        Feature values.

    Returns
    -------
    ndarray
        Sigmoid kernel SVM decision function gradient.
    """

    def kernel_gradient(features, support_vectors, coef0, gamma):
        """Compute the sigmoid kernel gradient."""

        return gamma*support_vectors*(
            1/cosh(gamma*dot(features, support_vectors.T) + coef0)**2)

    return sum(
        svm.dual_coef_[0, i]*kernel_gradient(
            features, svm.support_vectors_[i, :], svm.coef0, svm.gamma)
        for i, _ in enumerate(svm.support_vectors_)).reshape(-1)
