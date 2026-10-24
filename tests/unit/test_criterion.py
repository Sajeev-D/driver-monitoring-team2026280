import pytest

from src.driver_monitoring_ml.trees.criterion import information_gain

def test_information_gain():
    parent = [0,0,0,1,1]
    left = [0,0,1]
    right = [0,1]

    expected_information_gain = 1 / 75

    result = information_gain(parent, left, right)

    assert result == pytest.approx(expected_information_gain)