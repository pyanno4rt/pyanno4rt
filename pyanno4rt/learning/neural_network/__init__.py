"""
Neural network model module.

==================================================================

The module aims to provide methods and classes for modeling NTCP and TCP with \
neural network models.
"""

# Author: Tim Ortkamp

from ._neural_network_architectures import (
    build_vanilla_iocnn, build_vanilla_nn)
from ._neural_network import NeuralNetworkModel

__all__ = [
    'build_vanilla_iocnn',
    'build_vanilla_nn',
    'NeuralNetworkModel']
