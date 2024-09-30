"""Segment eigenvalues feature test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax.numpy import allclose, array, array_equal, identity, sort
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import SegmentEigenvalues

# %% Test definition


# Define the testcases
case_A = (array([identity(3), identity(3), identity(3)]), (1, 1, 1),
          (array([12.0, 6.0, 0.0]),
           array([[6., 0., 6.], [0., 6., 0.], [6., 0., 6.]])))


# Define the argument sets.
@mark.parametrize(
    'mask, spacing, expected',
    [case_A],
    ids=['case_A'])
def test_segment_eigenvalue(mask, spacing, expected):
    """Test the 'SegmentEigenvalues.compute' method."""

    # Assert the equality between actual and expected outcome
    assert allclose(sort(SegmentEigenvalues.compute(mask, spacing)[0]),
                    sort(expected[0]), atol=1e-6)
    assert array_equal(SegmentEigenvalues.compute(mask, spacing)[1],
                       expected[1])
