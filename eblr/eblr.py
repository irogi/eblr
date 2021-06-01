'''Explainable Boosted Linear Regression. Allows for the addition
of a number of hidden features to an arbitrary supervised learning model.

Classes
-------
EBLR
    Explainable Boosted Linear Regression
'''

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge

from .rpart_decision_tree import RpartDecisionTree

import logging

logger = logging.getLogger('eblr')
_MODELS = {
    'ridge': Ridge,
    'linear': LinearRegression,
    'bayesian': BayesianRidge
}


class EBLR:
    '''Explainable Boosted Linear Regression

    Boosts linear regression, with explainable, non-linear
    addititive features.


    Attributes
    -------
    num_feats: int, default = 50
        Number of additional paramaters to retrieve
    verbose: int, default = 0
        Verbosity. Higher means more verbose.
    initcp: float, default = 0.001
        The complexity parameter for tree pruning.
    base_model: str, default='ridge'
        The base linear model to learn. Options include 'ridge', 'linear', and 'bayesian'.
    kwargs: dict
        Keyword arguments for scikit-learn LinearRegression

    Methods
    -------
    fit: (X: np.ndarray, y: np.ndarray) => self
        Fits the model

    predict: (x: np.ndarray | int) => np.ndarray
        Make a future prediction

    explain: () => dict
        Generate explanation of additional regressors.
    '''

    def __init__(self, num_feats=50, verbose=0, initcp=0.001, base_model='ridge', **base_kwargs):
        self._base_model = base_model
        self._base_kwargs = base_kwargs
        self._init_cp = initcp
        self.model = _MODELS[self._base_model](**self._base_kwargs)

        self.conditions = []
        self._fi = []
        self.num_feats = num_feats
        self.verbose = verbose
        self.residuals = None

    @property
    def fi(self):
        return self._fi

    def fit(self, X, y):
        '''Fits the model, with given features and expected values.

        Attributes
        -------
        X: np.ndarray
            All the features
        y: np.ndarray
            The expected values given the features

        Returns
        ------
            (self, scores) Fitted model, with additional hidden features
        '''
        assert len(X) == len(y), 'Length of features does not match length of results'

        # Fit model
        self.model = _MODELS[self._base_model](**self._base_kwargs)
        self.conditions = []
        self._fi = []
        scores = self._fit_boosted(np.copy(X), np.copy(y))
        return self, scores

    def predict_intervals(self, X, intervals=(0.0, 0.5, 0.9)):
        '''Predicts future values quantiles

        Attributes
        -------
        X: np.ndarray | int
            Future prediction range, or covariates
        intervals: array-like floats, defaults to (0.0, 0.5, 0.9)
            An array-like list of floats.

        Returns
        -------
            Future quantile predictions
        '''
        predictions = self.predict(X).ravel()
        quants = set(0.5 + j*i/2 for i in intervals for j in (-1, 1))
        quants = sorted(list(quants))

        quantiles = [predictions + np.quantile(self.residuals, q) for q in quants]

        return np.array(quantiles)

    def predict(self, X):
        '''Predicts future values

        Attributes
        -------
        X: np.ndarray | int
            Future prediction range, or covariates

        Returns
        -------
            Future predictions
        '''
        X = self._add_feats(X)
        return self.model.predict(X)

    def explain(self):
        '''Generate an explanation of the hidden explanatory features'''
        return self.conditions

    def score(self, X, y):
        X = self._add_feats(X)
        return self.model.score(X, y)

    def _fit_boosted(self, X, y):
        '''Boosted fitting. Adds multiple additional explanatory features
        '''
        features = np.zeros(y.shape)
        self.model.fit(features, y)
        fi = []
        scores = []
        logger.debug('---------')
        for i in range(self.num_feats):
            scores.append(self.model.score(features, y))
            logger.debug(f'{i} score: {scores[0]}')
            residuals = self.model.predict(features) - y
            condition, feat = self._find_feat(X, residuals)
            if condition is None:
                break  # no more features to learn

            self.conditions.append(condition)

            if i == 0:
                # handle initial case
                features = feat.reshape(-1, 1).copy()
            else:
                features = np.c_[features, feat]
            self.model.fit(features, y)
            next_res = self.model.predict(features) - y
            fi.append(sum(np.abs(residuals-next_res))[0])
        features = np.c_[X, features].astype(np.float)
        self._generate_fi(fi, init_feats=X.shape[1], total_feats=features.shape[1])
        self.model.fit(features, y)
        self.residuals = self.model.predict(features) - y
        return scores

    def _find_feat(self, regressors, results):
        '''Find a hidden feature'''
        tree = RpartDecisionTree(self._init_cp)
        first_regressor_mtx = tree.biggest_residual_condition(regressors, results)

        if not len(first_regressor_mtx):  # unable to detect any features
            return None, None

        feat_mtx = np.array(first_regressor_mtx)[:, 1:]
        col = EBLR.create_regressor(regressors, feat_mtx)

        return feat_mtx, col

    def _add_feats(self, X):
        features = np.copy(X)
        # features = np.c_[np.array([], dtype=np.int64).reshape(len(X),0)]
        for condition in self.conditions:
            next_col = EBLR.create_regressor(X, condition)
            next_feat = np.array(next_col)
            features = np.c_[features, next_feat.reshape(-1, 1)]

        return features

    def _generate_fi(self, fi_res, init_feats, total_feats):
        assert len(fi_res) == len(self.conditions), 'Wrong number of feature importances'

        self._fi = [0] * total_feats
        for i in range(total_feats, init_feats, -1):
            feature_num = i-total_feats-1
            conds = self.conditions[feature_num]
            addition = self._fi.pop()
            for feat, *args in conds:
                self._fi[feat] += (fi_res[feature_num] + addition)
        self._fi = np.array(self._fi)
        self._fi /= sum(self._fi)

    @staticmethod
    def create_regressor(data, mtx):
        '''Creates regressor given specific matrix.
        '''
        num_dps = len(data)
        next_col = np.ones(num_dps, dtype=np.bool)

        for cond, threshold, comp_op in mtx:
            next_col *= comp_op(data[:, int(cond)], threshold)

        return next_col
