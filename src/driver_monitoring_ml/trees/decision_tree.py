from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root)) # put the project root in the import path

import numpy as np
from src.driver_monitoring_ml.trees.node import Node
from src.driver_monitoring_ml.trees.criterion import information_gain
import logging

logging.basicConfig(
    filename="outputs/logs/model.log",
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("in decision_tree.py")

class DecisionTreeClassifier:
    def __init__(
            self, 
            max_depth=10, 
            min_samples_split=2,
            max_features=None,
            random_state=None,
            root=None
    ):
        self.max_depth = max_depth # caps the max # of decision forks from root to leaf
        self.min_samples_split = min_samples_split # min # of samples for split to occur
        self.max_features = max_features # percentage (0-1) of feature columns evaluated when searching for best split
        self.random_state = random_state # seeds the random # generator to select different subset of features when building the tree
        self.root = root # root of the decision tree
        self.random_number_generator = np.random.RandomState(self.random_state)

    # function to build the decision tree recursively
    def fit(self, X: np.ndarray, y: np.ndarray):
        # convert X (n-dimensional array that stores the feature values) and y (1-dimensional array that stores the labels) to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # logging.info("building a decision tree")
        self.root = self._build_tree(X, y, depth=0)

        # return the decision tree classifier object
        return self

    def _build_tree(self, X, y, depth):
        num_samples = X.shape[0]
        num_features = X.shape[1]

        if depth >= self.max_depth or num_samples < self.min_samples_split or len(np.unique(y)) == 1:
            # value of the leaf node = the most common label in y
            leaf_value = self._majority_vote(y)
            return Node(value=leaf_value)

        # find the best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)

        # no split found
        if best_feature is None or best_gain <= 0:
            return Node(value=self._majority_vote(y))

        left_mask = X[:, best_feature] <= best_threshold # output is an array [True, False, ...]
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth+1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth+1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            gain=best_gain,
            left=left_child,
            right=right_child,
        )


    def _majority_vote(self, y):
        num_0s = 0
        num_1s = 0

        for label in y:
            if label == 0:
                num_0s += 1
            else:
                num_1s += 1

        if num_0s > num_1s:
            return 0
        else:
            return 1

    def _find_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        num_samples, num_features = X.shape

        # select a random subset of features to evaluate for the best split
        if self.max_features is None:
            features = range(num_features)
        else:             
            k = 0
            if self.max_features > 1 or self.max_features <= 0:
                raise ValueError("max_features must be in (0, 1]")
            else:
                k = max(1, round(self.max_features * num_features))

            features = self.random_number_generator.choice(num_features, k, replace=False) # randomly picks features to consider at this split

        for feature in features: # looping through all columns i.e. features
            thresholds = np.unique(X[:, feature]) # getting the array of unique values in the current column
            for threshold in thresholds: # treating each value in the column as a threshold
                left_indices = np.where(X[:, feature] <= threshold)[0] # left child indexes
                right_indices = np.where(X[:, feature] > threshold)[0] # right child indexes

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                left_labels = y[left_indices] # left child label array
                right_labels = y[right_indices] # right child label array

                gain = information_gain(y, left_labels, right_labels)

                if gain > best_gain: 
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    # Inference step: takes in new data, X i.e. the array of samples and gives back labels i.e. outputs
    def predict(self, X: np.ndarray):
        # logging.info("calling predict")
        X = np.asarray(X)
        outputs = []
        for sample in X:
            label = self._predict_step(sample, self.root) # traverses the tree for the current sample to get the output label
            outputs.append(label)

        return np.array(outputs)
    
    def _predict_step(self, x: np.ndarray, node: Node): # x is one row i.e. sample in X
        if node.value is not None: # check node is a leaf node
            return node.value
        if x[node.feature] <= node.threshold: # else traverse the correct fork based on input sample value
            return self._predict_step(x, node.left)
        return self._predict_step(x, node.right)
