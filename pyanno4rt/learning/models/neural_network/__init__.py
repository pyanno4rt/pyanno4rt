"""
Neural network module.

==================================================================

The module aims to provide methods and classes for neural network outcome \
modeling.
"""

# Author: Tim Ortkamp

from ._feed_forward_architectures import build_fnn, build_icnn
from ._feed_forward_net import FeedForwardNet

__all__ = [
    'build_fnn',
    'build_icnn',
    'FeedForwardNet']
