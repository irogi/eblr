import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

import pandas as pd
import numpy as np


class RpartDecisionTree():
    def __init__(self, initcp):
        pandas2ri.activate()

        ro.r('library(rpart)')
        ro.r('library(treeClust)')

        self.initialcp = initcp

    def biggest_residual_condition(self, regressors, results):
        '''Returns the condition for the biggest residual in the tree.
        If the biggest residual lies at the top of the tree, an empty array is returned
        '''
        df = pd.DataFrame(np.c_[results, regressors]).astype('float')
        df.columns = df.columns.map(str)
        df = df.rename(columns={'0': 'residual'})
        rdf = ro.conversion.py2rpy(df)

        tree = ro.r('rpart')('residual~.',
                             rdf,
                             control=ro.r('rpart.control')(cp=self.initialcp))
        cp_index = np.argmin(tree.rx2('cptable')[:, 3])  # '3' is xerror column index
        new_cp = tree.rx2('cptable')[cp_index, 0]        # '0' is cp column index
        tree = ro.r('prune')(tree, new_cp)

        nodes = ro.r('rpart.predict.leaves')(tree, rdf, type='where')
        df['nodes'] = nodes
        leaf_where = abs(df.groupby('nodes').agg('mean')['residual']).idxmax()
        leaf = int(tree.rx2('frame').iloc[leaf_where-1].name)

        rules = ro.r('path.rpart')(tree, node=leaf)[0]

        path = []
        for rule in rules[1:]:  # skip root node
            if '>=' in rule:
                comp_op = np.greater_equal
                string_operator = '>='
            elif '>' in rule:
                comp_op = np.greater
                string_operator = '>'
            elif '<=' in rule:
                comp_op = np.less_equal
                string_operator = '<='
            elif '<' in rule:
                comp_op = np.less
                string_operator = '<'
            else:
                raise Exception('Unknown operator')
            feature, threshold = rule.split(string_operator)

            # -1 to offset from residuals column
            path.append((-1, int(feature) - 1, float(threshold), comp_op))
        return path
