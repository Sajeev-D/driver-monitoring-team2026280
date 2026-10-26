from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root)) # put the project root in the import path

from src.driver_monitoring_ml.trees.decision_tree import DecisionTreeClassifier
import numpy as np 
import logging
from typing import List

logging.basicConfig(
    filename="outputs/logs/model.log",
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("in random_forest.py")

class RandomForestClassifier:
    def __init__(
            self,
            number_of_trees=100,
            max_tree_depth=10,
            min_samples_split=2,
            max_features=None,
            max_samples=1.0,
            random_state=None
    ):
        self.number_of_trees = number_of_trees # Number of trees in the forest
        self.max_tree_depth = max_tree_depth # caps the max # of decision forks from root to leaf
        self.min_samples_split = min_samples_split # min # of samples for split to occur
        self.max_features = max_features # percentage (0-1) of feature columns evaluated when searching for best split
        self.max_samples = max_samples # Fraction of rows to draw per bootstrap sample
        self.random_state = random_state # seeds the random # generator to select different subset of features when building the tree

        self.trees: List[DecisionTreeClassifier] = [] # list to store DecisionTreeClassifier objects i.e. decision trees
        self.random_number_generator = np.random.RandomState(self.random_state) 

    # function to create the forest
    def fit(self, X: np.ndarray, y:np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)
        num_samples = X.shape[0] # number of rows

        self.trees = []

        sample_size_per_tree = int(num_samples * self.max_samples)

        num_trees_to_generate = self.number_of_trees

        for i in range(num_trees_to_generate):
            # get bootstrap dataset by choosing self.max_samples # of rows from the original dataset w/ replacement
            bootstrap_row_indexes = self.random_number_generator.choice(num_samples, sample_size_per_tree, replace=True)
            X_bootstrap = X[bootstrap_row_indexes]
            y_bootstrap = y[bootstrap_row_indexes]

            # generate a random seed for each tree so that all trees do not start from the same seed
            tree_seed = self.random_number_generator.randint(0, 2**31 - 1)

            # create the decision tree and append to list
            tree = DecisionTreeClassifier(
                max_depth=self.max_tree_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=tree_seed
            )
            tree.fit(X_bootstrap, y_bootstrap)

            self.trees.append(tree)

        return self

    # generate the output array for a list of input samples using the random forest
    def predict(self, X: np.ndarray):
        if len(self.trees) == 0:
            logging.error("predict called before fit")
            raise RuntimeError("The forest has not been fitted yet. Call fit first.")

        X = np.asarray(X)
        outputs = []

        for tree in self.trees:
           # get output labels
           outputs.append(tree.predict(X))

        outputs = np.array(outputs)
        vote_fraction = outputs.mean(axis=0) # fraction of trees that voted 1, per column
        final_labels = vote_fraction > 0.5 # True where more than half voted 1

        return final_labels.astype(int) # convert True/False array elements to 1/0 int