#!/usr/bin/env python3
# Create by: Will McGrath
"""
Custom transformer for pipeline

Input: Categorical data

Description: Will create new binary columns for categ data

Output: Transformed data
"""
from sklearn.base import BaseEstimator


class OutletTypeEncoder(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, documents, y=None):
        # replace NaN vals
        return self

    def transform(self, X_dataset):
        # encode categ data like topic and day of week feats
        return X_dataset
