#!/usr/bin/env python3
# BDA 696 Final Project
# Create by: Will McGrath
import os
import sys

import _pickle as cPickle
import pandas as pd


class FinalModel(object):
    def __init__(self):
        # load the non-labeled data with feats and labeled data
        self.this_dir = os.path.dirname(os.path.realpath(__file__))

    def load_pipeline(self, path):
        # load pipeline
        model_dir = os.path.join(self.this_dir, path)
        with open(model_dir, "rb") as f:
            pipeline = cPickle.load(f)

        return pipeline

    def load_data(self):
        # load the merged df with old data
        data_dir = os.path.join(self.this_dir, "../data")
        nulls_merged_data = os.path.join(
            data_dir, "merged_nulls_labeled_feat_tweets.csv"
        )

        # open data as dataframe
        nulls_merged_df = pd.read_csv(nulls_merged_data, sep=",")

        # final dataset shape before training
        print(
            f"Dataset:[{nulls_merged_df.shape[0]} rows {nulls_merged_df.shape[1]} cols]"
        )

        return nulls_merged_df

    def predict_data(self, df, model):
        # create X_test
        X_test = df.iloc[:, :-1]

        # set predictions as target
        df.iloc[:, -1] = model.predict(X_test)

        return df

    def main(self):
        # load model
        xgb_pipeline = self.load_pipeline("../models/xgb_pipeline.pickle")

        # load dataset
        nulls_merged_df = self.load_data()

        # predict target
        nulls_merged_df = self.predict_data(nulls_merged_df, xgb_pipeline["model"])

        # response variable distr
        print(f"Target Value Count: \n {nulls_merged_df['target'].value_counts()}")
        print(f"Target N/A Count: {nulls_merged_df['target'].isna().sum()}")


if __name__ == "__main__":
    sys.exit(FinalModel().main())
