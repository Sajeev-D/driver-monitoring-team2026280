from src.driver_monitoring_ml.trees.node import Node

def test_node_initialization():
    node = Node(feature=0, threshold=1.5, gain=0.5, left=None, right=None, value=1)

    assert node.feature == 0
    assert node.threshold == 1.5
    assert node.gain == 0.5
    assert node.left is None
    assert node.right is None
    assert node.value == 1

def test_is_leaf_node():
    leaf_node = Node(value=1)
    non_leaf_node = Node()

    assert True == leaf_node.is_leaf_node()
    assert False == non_leaf_node.is_leaf_node()