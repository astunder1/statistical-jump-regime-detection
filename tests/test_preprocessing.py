import numpy as np
import pandas as pd
import pytest

from regimejump.preprocessing import standardize_from_training_window


def test_standardization_uses_training_statistics():
    training = pd.DataFrame(
        {
            "a": [0.0, 2.0],
            "b": [10.0, 14.0],
        }
    )

    data = pd.DataFrame(
        {
            "a": [3.0],
            "b": [16.0],
        }
    )

    result = standardize_from_training_window(training, data)

    expected = pd.DataFrame(
        {
            "a": [2.0],  # (3 - 1) / 1
            "b": [2.0],  # (16 - 12) / 2
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_training_standardizes_to_zero_mean_and_unit_variance():
    training = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0],
            "b": [10.0, 12.0, 14.0],
        }
    )

    result = standardize_from_training_window(training, training)

    np.testing.assert_allclose(result.mean().to_numpy(), 0.0)
    np.testing.assert_allclose(
        result.std(ddof=0).to_numpy(),
        1.0,
    )


def test_future_data_does_not_affect_earlier_transformations():
    training = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    data = pd.DataFrame({"a": [3.0, 4.0, 5.0]})

    original = standardize_from_training_window(training, data)

    changed = data.copy()
    changed.iloc[2] = 1000.0
    perturbed = standardize_from_training_window(training, changed)

    pd.testing.assert_frame_equal(
        original.iloc[:2],
        perturbed.iloc[:2],
    )


def test_rejects_mismatched_columns():
    training = pd.DataFrame({"a": [0.0, 1.0]})
    data = pd.DataFrame({"b": [0.0, 1.0]})

    with pytest.raises(ValueError, match="identical columns"):
        standardize_from_training_window(training, data)


def test_rejects_nonfinite_training_data():
    training = pd.DataFrame({"a": [0.0, np.nan]})
    data = pd.DataFrame({"a": [1.0]})

    with pytest.raises(ValueError, match="finite"):
        standardize_from_training_window(training, data)
