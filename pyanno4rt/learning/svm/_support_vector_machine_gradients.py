"""Support vector machine model gradients."""

# Author: Tim Ortkamp

# %% External package import

from numpy import cosh, dot, exp

# %% Input gradients


def platt_gradient(svm, features):
    """
    Compute the gradient of the Platt scaling function.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the pre-fitted prediction model.

    features : ndarray
        Values of the input features.

    Returns
    -------
    ndarray
        Gradient of the Platt scaling function.
    """

    # Get the prediction value
    prediction = svm.predict_proba(features)[:, 1]

    return -svm.probA_[0]*prediction*(1-prediction)


def linear_gradient(svm, features):
    """
    Compute the gradient of the linear kernel SVM.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the pre-fitted prediction model.

    features : ndarray
        Values of the input features.

    Returns
    -------
    ndarray
        Gradient of the linear kernel SVM.
    """

    return platt_gradient(svm, features)*svm.coef_.reshape(-1)


def rbf_gradient(svm, features):
    """
    Compute the gradient of the radial basis function kernel SVM.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the pre-fitted prediction model.

    features : ndarray
        Values of the input features.

    Returns
    -------
    ndarray
        Gradient of the radial basis function kernel SVM.
    """

    def kernel_gradient(features, support_vectors, gamma):
        """Compute the radial basis kernel gradient."""

        return -2*gamma*(features-support_vectors)*exp(-gamma*(
            dot(features, features.T)-2*dot(features, support_vectors.T)
            + dot(support_vectors, support_vectors.T)))

    return platt_gradient(svm, features) * sum(
        svm.dual_coef_[0, i]*kernel_gradient(
            features, svm.support_vectors_[i, :], svm.gamma)
        for i, _ in enumerate(svm.support_vectors_))[0].reshape(-1)


def poly_gradient(svm, features):
    """
    Compute the gradient of the polynomial kernel SVM.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the pre-fitted prediction model.

    features : ndarray
        Values of the input features.

    Returns
    -------
    ndarray
        Gradient of the polynomial kernel SVM.
    """

    def kernel_gradient(features, support_vectors, coef0, degree, gamma):
        """Compute the polynomial kernel gradient."""

        return gamma*degree*support_vectors*(
            gamma*dot(features, support_vectors.T) + coef0)**(degree-1)

    return platt_gradient(svm, features) * sum(
        svm.dual_coef_[0, i]*kernel_gradient(
            features, svm.support_vectors_[i, :], svm.coef0, svm.degree,
            svm.gamma)
        for i, _ in enumerate(svm.support_vectors_)).reshape(-1)


def sigmoid_gradient(svm, features):
    """
    Compute the gradient of the sigmoid kernel SVM.

    Parameters
    ----------
    svm : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the pre-fitted prediction model.

    features : ndarray
        Values of the input features.

    Returns
    -------
    ndarray
        Gradient of the sigmoid kernel SVM.
    """

    def kernel_gradient(features, support_vectors, coef0, gamma):
        """Compute the sigmoid kernel gradient."""

        return gamma*support_vectors*(
            1/cosh(gamma*dot(features, support_vectors.T) + coef0)**2)

    return platt_gradient(svm, features) * sum(
        svm.dual_coef_[0, i]*kernel_gradient(
            features, svm.support_vectors_[i, :], svm.coef0, svm.gamma)
        for i, _ in enumerate(svm.support_vectors_)).reshape(-1)
