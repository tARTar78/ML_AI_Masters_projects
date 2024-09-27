from typing import Optional

import numpy as np

from sem_dt_rf.decision_tree.criterio import Criterion, GiniCriterion, EntropyCriterion, MSECriterion
from sem_dt_rf.decision_tree.tree_node import TreeNode


class DecisionTree:
    def __init__(self, max_depth: int = 10, min_leaf_size: int = 5, min_improvement: Optional[float] = None):
        self.criterion = None
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.min_improvement = min_improvement
        
    def _build_nodes(self, x: np.ndarray, y: np.ndarray, criterion: Criterion, indices: np.ndarray, node: TreeNode):
        """
        Builds tree recursively

        Parameters
        ----------
        x : samples in node, np.ndarray.shape = (n_samples, n_features)
        y : target values, np.ndarray.shape = (n_samples, )
        criterion : criterion to split by, Criterion
        indices : samples' indices in node,
            np.ndarray.shape = (n_samples, )
            nd.ndarray.dtype = int
        node : current node to split, TreeNode
        """
        if self.max_depth is not None and self.max_depth <= node.depth:
            return
        if self.min_leaf_size is not None and self.min_leaf_size >= len(indices):
            return node
        if np.unique(y[indices]).shape[0] <= 1:
            return node

        q, threshold, feature = node.get_best_split(x[indices], y[indices], criterion)
        if self.min_improvement is not None and self.min_improvement >= q:
            return node
        mask, left_child, right_child = node.split(
            x[indices], y[indices], criterion,
            feature=feature,
            threshold=threshold,
            improvement=q
        )
        self._build_nodes(x, y, criterion, indices[~mask], left_child)
        self._build_nodes(x, y, criterion, indices[mask], right_child)

    def _get_nodes_predictions(self, x: np.ndarray, predictions: np.ndarray, indices: np.ndarray, node: TreeNode):
        if node.is_terminal:
            predictions[indices] = node.predict_val
            return
        mask = node.get_best_split_mask(x)
        mask = mask[indices]
        self._get_nodes_predictions(x, predictions, indices[~mask], node.child_left)
        self._get_nodes_predictions(x, predictions, indices[mask], node.child_right)

    def _get_nodes_q(self, qf: np.ndarray, node: TreeNode):
        qf[node.feature] = node.impurity
        if node.is_terminal:
            return
        self._get_nodes_q(qf, node.child_left)
        self._get_nodes_q(qf, node.child_right)
        
class ClassificationDecisionTree(DecisionTree):
    def __init__(self, criterion: str = "gini", **kwargs):
        super().__init__(**kwargs)

        if criterion not in ["gini", "entropy", "mse"]:
            raise ValueError('Unsupported criterion', criterion)
        self.criterion = criterion
        self.n_classes = 0
        self.n_features = 0
        self.root = None

    def fit(self, x, y):
        self.n_classes = np.unique(y).shape[0]
        self.n_features = x.shape[1]
        if self.criterion == "gini":
            criterion = GiniCriterion(self.n_classes)
        elif self.criterion == "entropy":
            criterion = EntropyCriterion(self.n_classes)
        elif self.criterion == "mse":
            criterion = MSECriterion()
        self.root = TreeNode(
            depth=0,
            predict_val=criterion.get_predict_val(y),
            impurity=criterion.score(y)
        )
        self._build_nodes(x, y, criterion, np.arange(x.shape[0]), self.root)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)
	
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        pred = np.zeros((x.shape[0], self.n_classes))
        self._get_nodes_predictions(x, pred, np.arange(x.shape[0]), self.root)
        return pred

    @property
    def feature_importance_(self) -> np.ndarray:
        """
        Returns
        -------
        importance : cumulative improvement per feature, np.ndarray.shape = (n_features, )
        """
        return self.get_q_array().argsort(axis=0)
    def get_q_array(self) -> np.ndarray:
        qf = np.zeros(self.n_features)
        self._get_nodes_q(qf, self.root)
        return qf
        
class RegressionDecisionTree(DecisionTree):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.n_features = 0
        self.root = None

    def fit(self, x, y):
        self.n_features = x.shape[1]
        criterion = MSECriterion()
        self.root = TreeNode(
            depth=0,
            predict_val=criterion.get_predict_val(y),
            impurity=criterion.score(y)
        )
        self._build_nodes(x, y, criterion, np.arange(x.shape[0]), self.root)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = np.zeros(x.shape[0])
        self._get_nodes_predictions(x, pred, np.arange(x.shape[0]), self.root)
        return pred
