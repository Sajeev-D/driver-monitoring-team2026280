import numpy as np
import pytest

from src.driver_monitoring_ml.trees.decision_tree import DecisionTreeClassifier
from src.driver_monitoring_ml.trees.node import Node

# checking if hyperparameter initialization works
def test_initialization_sets_parameters_and_root():
    root = Node(value=0)
    classifier = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=3,
        max_features=0.5,
        random_state=7,
        root=root,
    )

    assert classifier.max_depth == 4
    assert classifier.min_samples_split == 3
    assert classifier.max_features == 0.5
    assert classifier.random_state == 7
    assert classifier.root is root


def test_fit_builds_tree_and_returns_classifier():
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    classifier = DecisionTreeClassifier()

    result = classifier.fit(X, y)

    assert result is classifier
    assert classifier.root.feature == 0
    assert classifier.root.threshold == 1
    assert classifier.root.left.value == 0
    assert classifier.root.right.value == 1


def test_build_tree_stops_at_max_depth():
    classifier = DecisionTreeClassifier(max_depth=0)

    root = classifier._build_tree(
        np.array([[0], [1]]),
        np.array([0, 1]),
        depth=0,
    )

    assert root.value == 1


def test_build_tree_stops_when_sample_count_is_below_minimum():
    classifier = DecisionTreeClassifier(min_samples_split=3)

    root = classifier._build_tree(
        np.array([[0], [1]]),
        np.array([0, 1]),
        depth=0,
    )

    assert root.value == 1


def test_build_tree_stops_for_pure_labels():
    classifier = DecisionTreeClassifier()

    root = classifier._build_tree(
        np.array([[0], [1]]),
        np.array([0, 0]),
        depth=0,
    )

    assert root.value == 0


def test_build_tree_returns_leaf_when_no_split_is_available():
    classifier = DecisionTreeClassifier()

    root = classifier._build_tree(
        np.array([[1], [1]]),
        np.array([0, 1]),
        depth=0,
    )

    assert root.value == 1


def test_build_tree_returns_leaf_when_best_split_has_no_gain():
    classifier = DecisionTreeClassifier()

    root = classifier._build_tree(
        np.array([[0], [0], [1], [1]]),
        np.array([0, 1, 0, 1]),
        depth=0,
    )

    assert root.value == 1

# test if the _majority_vote function work correctly
def test_majority_vote_returns_most_common_binary_label():
    classifier = DecisionTreeClassifier()

    assert classifier._majority_vote([0, 0, 1]) == 0
    assert classifier._majority_vote([0, 1, 1]) == 1

# test _majority_vote function when there is tie in the # of labels
def test_majority_vote_returns_one_on_tie():
    classifier = DecisionTreeClassifier()

    assert classifier._majority_vote([0, 1]) == 1


def test_find_best_split_returns_split_with_highest_gain():
    X = np.array([[0], [1], [2], [3]]) # this is one column of data e.g. feature 0 column
    y = np.array([0, 0, 1, 1]) # these are the labels for each row
    classifier = DecisionTreeClassifier()

    feature, threshold, gain = classifier._find_best_split(X, y)

    """
    when threshold is 1, 
    * left child samples with values 0,1 which has labels [0,0] => 100% pure therefore Gini = 0
    * right child samples with values 2,3 which has labels [1,1] => 100% pure therefore Gini = 0
    * for the parent node i.e. X, it's Gini = 0.5 therefore Info_gain = 0.5 - 0 - 0 = 0.5 
    * 0.5 is the max Info_gain for a split at this node.
    * Recall, we want to maximise Info_gain = I_p - [ (n_left / n) * I_l + (n_right / n) * I_r ] at each split
    """
    assert feature == 0
    assert threshold == 1
    assert gain == 0.5


def test_find_best_split_returns_no_split_for_constant_feature():
    X = np.array([[1], [1]]) # both samples have the same value, 1 so 1 is the only threshold used to find a split
    y = np.array([0, 1]) # but the feature gives no way to tell the 2 samples apart
    classifier = DecisionTreeClassifier()

    feature, threshold, gain = classifier._find_best_split(X, y)

    # so the value of best_feature, best_threshold, and best_gain still hold their starting values
    assert feature is None
    assert threshold is None
    assert gain == -1


@pytest.mark.parametrize("max_features", [-0.1, 0, 1.1]) # runs the test case 3 times i.e. once for each parameter value
def test_find_best_split_raises_for_max_features_out_of_bounds(max_features):
    classifier = DecisionTreeClassifier(max_features=-0.1)

    # If the code raises a ValueError whose message contains max_features, the test passes.
    with pytest.raises(ValueError, match="max_features"): # expect to raise an error for all three parameter valeus
        classifier._find_best_split(
            np.array([[0], [1]]),
            np.array([0, 1]),
        )


def test_find_best_split_uses_seeded_feature_subset():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([0, 1, 0, 1])
    classifier = DecisionTreeClassifier(max_features=0.5, random_state=0)

    feature, threshold, gain = classifier._find_best_split(X, y)

    """
    With random_state=0, the random number generator is seeded to pick Feature 1. 
    Recall: The same seed with the same generator type and the same calls always gives the same result.
    
    Since Feature 1 matches y perfectly, 
    it splits on Feature 1 at threshold 0 with gain 0.5.
    """

    assert feature == 1
    assert threshold == 0
    assert gain == 0.5

# checking leaf node 
def test_predict_step_traverses_left_and_right_branches():
    # build a small tree manually
    root = Node(
        feature=0,
        threshold=1.5,
        left=Node(value=0),
        right=Node(value=1),
    )
    classifier = DecisionTreeClassifier(root=root)

    assert classifier._predict_step(np.array([1]), root) == 0 # 1 <= 1.5, goes left, returns 0
    assert classifier._predict_step(np.array([2]), root) == 1 # 2 > 1.5, goes right, returns 1


def test_predict_returns_labels_for_input_samples():
    classifier = DecisionTreeClassifier()

    # build a tree using fit function
    classifier.fit([[0], [1], [2], [3]], [0, 0, 1, 1])

    predictions = classifier.predict([[0], [3]]) # requesting label for 2 input samples, [0] and [3]

    np.testing.assert_array_equal(predictions, np.array([0, 1])) # assert_array_equal compares the two arrays element by element