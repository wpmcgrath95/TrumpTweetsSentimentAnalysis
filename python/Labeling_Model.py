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
        # load the non-labeled data with feats and labeled data (adds target)
        data_dir = os.path.join(self.this_dir, "../data")
        feat_data_dir = os.path.join(data_dir, "twitter_data_with_feats.csv")
        labeled_data_dir = os.path.join(data_dir, "labeled_realdonaldtrump.csv")

        # open data as dataframe
        feat_data_df = pd.read_csv(feat_data_dir, sep=",")
        labeled_data_df = pd.read_csv(labeled_data_dir, sep=",")
        labeled_data_df = labeled_data_df[["link", "target"]]

        # merge dataframes
        merged_df = pd.merge(feat_data_df, labeled_data_df, how="inner", on=["link"])

        # drop target rows with NaN or missing vals (non-labeled)
        merged_df = merged_df[pd.notnull(merged_df["target"])].reset_index(drop=True)

        # drop all unusable cols
        drop_cols = ["no_hr_date", "id", "link", "content", "date", "processed_content"]
        merged_df.drop(drop_cols, axis=1, inplace=True)

        return merged_df

    def upsample(self):
        # upsample class distribution
        pass

    def train_test_split(self):
        # drop non-used cols for training

        # train and test split

        pass

    def train(self, df):
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

    def baseline_model(self):
        # create dummy model to pred sentiment to compare real model to
        # don't need
        pass

    def performance(self):
        # ROC curve, PPV, TPR, F1, SHAP values
        # test multicorrelation using VIF
        pass

    def main(self):
        # create merged dataset
        merged_df = self.merge_data()

        # final dataset shape before training
        print(f"Dataset:[{merged_df.shape[0]} rows x {merged_df.shape[1]} cols]")

        # response variable distr
        print(f"Target Value Count: \n {merged_df['target'].value_counts()}")
        print(f"Target N/A Count: {merged_df['target'].isna().sum()}")

        # save merged df with feats and target to csv in data folder
        merged_dir = os.path.join(
            self.this_dir, "../data/merged_labeled_feat_tweets.csv"
        )
        merged_df.to_csv(merged_dir, encoding="utf-8", index=False)

        return None


if __name__ == "__main__":
    sys.exit(LabelingModel().main())
