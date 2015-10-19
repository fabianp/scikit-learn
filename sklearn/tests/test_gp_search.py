
import sys
import numpy as np
from sklearn.externals.six.moves import cStringIO as StringIO
from sklearn.gp_search import GPSearchCV
from sklearn.utils.testing import assert_true, assert_equal, assert_raises

# A toy dataset for the tests
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])


class MockClassifier(object):
    """Dummy classifier to test the cross-validation"""
    def __init__(self, foo_param=0):
        self.foo_param = foo_param

    def fit(self, X, Y):
        assert_true(len(X) == len(Y))
        return self

    def predict(self, T):
        return T.shape[0]

    predict_proba = predict
    decision_function = predict
    transform = predict

    def score(self, X=None, Y=None):
        if self.foo_param > 1:
            score = 1.
        else:
            score = 0.
        return score

    def get_params(self, deep=False):
        return {'foo_param': self.foo_param}

    def set_params(self, **params):
        self.foo_param = params['foo_param']
        return self


def test_gp_search():
    clf = MockClassifier()
    gp_search = GPSearchCV(clf, {'foo_param': ['int', [1, 3]]}, verbose=3)
    # make sure it selects the smallest parameter in case of ties
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    gp_search.fit(X, y)
    sys.stdout = old_stdout
    assert_equal(gp_search.best_estimator_.foo_param, 2)

    for i, foo_i in enumerate([1, 2, 3]):
        assert_true(gp_search.scores_[i][0]
                    == {'foo_param': foo_i})
    # Smoke test the score etc:
    gp_search.score(X, y)
    gp_search.predict_proba(X)
    gp_search.decision_function(X)
    gp_search.transform(X)

    # Test exception handling on scoring
    gp_search.scoring = 'sklearn'
    assert_raises(ValueError, gp_search.fit, X, y)

