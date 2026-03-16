import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn_autobatch.estimators.auto_batch_mixin import AutoBatchMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import check_estimator

class KNNBatchClassifier(AutoBatchMixin, KNeighborsClassifier):
    pass

class LogisticBatchClassifier(AutoBatchMixin, LogisticRegression):
    pass

class MLPBatchClassifier(AutoBatchMixin, MLPClassifier):
    pass

class SVCBatchClassifier(AutoBatchMixin, SVC):
    pass


class TestAutoBatchMixin(unittest.TestCase):

    def get_estimators(self):
        return {
            "KNN": KNNBatchClassifier(),
            "Logistic": LogisticBatchClassifier(),
            "MLP": MLPBatchClassifier(),
            "SVC": SVCBatchClassifier(),
        }
    
    def test_sklearn(self):

        for name, clf in self.get_estimators().items():
            with self.subTest(estimator=name):
                check_estimator(clf)

    def test_iris(self):
        X, y = load_iris(return_X_y=True)
        n_classes = len(np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0
        )

        for name, clf in self.get_estimators().items():
            with self.subTest(estimator=name):
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                metric_val = cohen_kappa_score(y_test, y_pred)
                self.assertTrue(metric_val > 0, "Classifier should be better than random!")

                if hasattr(clf, "predict_proba"):
                    probas = clf.predict_proba(X)

                    self.assertIsNotNone(probas, "Probabilites are None")
                    self.assertFalse(np.isnan(probas).any(), "NaNs in probability predictions")
                    self.assertFalse(np.isinf(probas).any(), "Inf in probability predictions")
                    self.assertTrue(
                        probas.shape[0] == X.shape[0],
                        "Different number of objects in prediction",
                    )
                    self.assertTrue(
                        probas.shape[1] == n_classes,
                        "Wrong number of classes in proba prediction",
                    )
                    self.assertTrue(np.all(probas >= 0), "Negative probabilities")
                    self.assertTrue(np.all(probas <= 1), "Probabilities bigger than one")
                    
                    prob_sums = np.sum(probas, axis=1)
                    self.assertTrue(
                        np.allclose(prob_sums, np.asanyarray([1 for _ in range(X.shape[0])])),
                        "Not all sums close to one",
                    )