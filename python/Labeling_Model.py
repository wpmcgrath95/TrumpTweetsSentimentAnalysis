#!/usr/bin/env python3
# BDA 696 Final Project
# Create by: Will McGrath

"""
1. choose 100 random tweets and label them
   Note: called labeled_trump_twitter_data.csv
2. build a model on those 100 random tweets
   Note: combine labeled_trump_twitter_data.csv with
         twitter_data_with_feats.csv and then get rid of
         non-labeled tweets
3. use this model to predict the rest of the tweets

Note: SHOULD I REDO FEATS HERE B/C WHEN I CUT OFF THE MID PART THE NUMS IN FEATS
      ARE RELATED TO DATA W/O CUTTOFF??
"""
import os
import sys

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


class LabelingModel(object):
    def __init__(self):
        # load the non-labeled data with feats and labeled data
        self.this_dir = os.path.dirname(os.path.realpath(__file__))

    def merge_data(self):
        # load the non-labeled data with feats and labeled data
        data_dir = os.path.join(self.this_dir, "../data")
        feat_data_dir = os.path.join(data_dir, "twitter_data_with_feats.csv")
        labeled_data_dir = os.path.join(data_dir, "labeled_realdonaldtrump.csv")

        # open data as dataframe
        feat_data_df = pd.read_csv(feat_data_dir, sep=",")
        labeled_data_df = pd.read_csv(labeled_data_dir, sep=",")
        labeled_data_df = labeled_data_df[["link", "target"]]

        # merge dataframes
        merged_df = pd.merge(feat_data_df, labeled_data_df, how="inner", on=["link"])

        # drop rows with NaN
        merged_df = merged_df[pd.notnull(merged_df["target"])].reset_index(drop=True)

        return merged_df

    def encoding(self):
        # transformating categ data
        # need to encode topic and day of week feats
        pass

    def upsample(self):
        # upsample class distribution
        pass

    def train(self, df):
        # drop ids
        # define X = predictors and y = response vars
        X = df.iloc[:, 2:-1]
        y = df.iloc[:, -1]

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        # xgboost model
        xgb_model = xgb.XGBClassifier()

        return xgb_model

    def pipeline(self):
        # add gridsearch to maximize recall
        pass

    def main(self):
        merged_df = self.merge_data()
        print(merged_df)
        print(merged_df["target"].value_counts())
        print(merged_df["target"].isna().sum())

        # save merged df with feats and target to csv in data folder
        merged_dir = os.path.join(
            self.this_dir, "../data/merged_labeled_feat_tweets.csv"
        )
        merged_df.to_csv(merged_dir, encoding="utf-8", index=False)

        return None


if __name__ == "__main__":
    sys.exit(LabelingModel().main())
