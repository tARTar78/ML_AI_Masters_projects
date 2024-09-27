import numpy as np
import unittest


from sem_dt_rf.decision_tree.tree_node import TreeNode
from sem_dt_rf.decision_tree.criterio import GiniCriterion


class TestTreeNode(unittest.TestCase):
    def test_get_best_split(self):
        n, m = 100, 10
        f = 3
        criterion = GiniCriterion(n_classes=2)
        x = np.c_[[np.arange(n) if f == fi else np.random.random(size=n) for fi in range(m)]].T
        num_zeros = n // 2
        y = np.r_[np.ones(num_zeros), np.zeros(n - num_zeros)].astype(int)
        improvement, best_threshold, best_feature = TreeNode.get_best_split(x, y, criterion)
        assert np.isclose(best_threshold, 49.5)
        assert np.isclose(improvement, 0.5)
        assert np.isclose(best_feature, f)
