class Node:
    def __init__(self, feature=None, threshold=None, gain=None, left=None, right=None, value=None):
        self.feature = feature # column index of the feature that this node splits on
        self.threshold = threshold # the threshold value that this node splits on
        self.gain = gain # the information gain of this node
        self.left = left # the left child node
        self.right = right # the right child node
        self.value = value # the value of the node (for leaf nodes)

    def is_leaf_node(self):
        # only leaf nodes have a value
        if self.value is not None:
            return True
        else:
            return False