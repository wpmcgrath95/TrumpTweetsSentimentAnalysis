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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
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

        # drop non-used cols for training
        drop_cols = ["no_hr_date", "id", "link", "content", "date", "processed_content"]
        merged_df.drop(drop_cols, axis=1, inplace=True)

        return merged_df

    def save_data(self, df, path):
        # save data to csv in a folder
        df_dir = os.path.join(self.this_dir, path)
        df.to_csv(df_dir, encoding="utf-8", index=False)

        return None

    def upsample(self):
        # upsample class distribution
        pass

    def train_test_split(self, df):
        # train and test split
        # define X = predictors and y = response var
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def baseline_model(self):
        # create dummy model to pred sentiment to compare real model to
        # don't need
        pass

    def performance(self, model, X_train, X_test, y_test):
        # predict target on train and test set
        predict_train = model.predict(X_train)
        predict_test = model.predict(X_test)

        # get accuracy
        accuracy = model.score(X_test, y_test)
        print(predict_train)
        print(predict_test)

        # ROC curve, PPV, TPR, F1, SHAP values
        # test multicorrelation using VIF

        return accuracy

    def feat_importance(self, model, X_test):
        # feature importance
        feat_impt = {}  # dict to hold feature_name: feature_importance
        for feat, impt in zip(X_test, model.feature_importances_):
            feat_impt[feat] = impt

        if isinstance(model, RandomForestClassifier):
            feat_impt_df = pd.DataFrame.from_dict(feat_impt, orient="index").rename(
                columns={0: "Gini-importance"}
            )
            # feat_impt_df.sort_values(by='Gini-importance').plot(kind='bar', rot=45)
            feat_impt_df = feat_impt_df["Gini-importance"].sort_values(ascending=False)

        else:
            feat_impt_df = pd.DataFrame.from_dict(feat_impt, orient="index").rename(
                columns={0: "Gain"}
            )

            # a = os.path.join(self.this_dir, "../plots/xgb_feat_impt.png")
            # fig = feat_impt_df.sort_values(by='Gain').plot(kind='bar', rot=45)
            # fig.get_figure()
            # fig.savefig(a)
            # feat_impt_df.sort_values(by='Gain').plot(kind='bar', rot=45)
            feat_impt_df = feat_impt_df["Gain"].sort_values(ascending=False)

        return feat_impt_df

    def corr_matrix(self, df):
        # only shows correlation between numerical features
        correlations = df.corr(method="pearson")

        y_name = "target"
        correlations.sort_values(y_name, ascending=False)
        fig, ax = plt.subplots(figsize=(60, 60))
        sns.heatmap(
            correlations,
            vmax=1.0,
            center=0,
            fmt=".2f",
            square=True,
            linewidths=0.3,
            annot=True,
            cbar_kws={"shrink": 0.80},
            cmap="YlGnBu",
        )
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.5  # Add 0.5 to the bottom
        t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values

        # save figure
        corr_matrix_plt = os.path.join(self.this_dir, "../plots/corr_matrix.png")
        fig.savefig(corr_matrix_plt, bbox_inches="tight")

        return None

    def main(self):
        # set seed
        np.random.seed(seed=1)

        # create merged dataset with feats and target
        merged_df = self.merge_data()

        # save merged dataset to csv
        data_path = "../data/merged_labeled_feat_tweets.csv"
        self.save_data(merged_df, data_path)

        # final dataset shape before training
        print(f"Dataset:[{merged_df.shape[0]} rows x {merged_df.shape[1]} cols]")

        # response variable distr
        print(f"Target Value Count: \n {merged_df['target'].value_counts()}")
        print(f"Target N/A Count: {merged_df['target'].isna().sum()}")

        # split the data into train and test set
        X_train, X_test, y_train, y_test = self.train_test_split(merged_df)

        # create RandomForest and XGBoost models
        rf_model = RandomForestClassifier(random_state=0)
        xgb_model = xgb.XGBClassifier(
            max_depth=4, eta=0.3, colsample_bytree=0.1, n_estimators=100, seed=0
        )

        # fit models with training data
        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)

        # RandomForest model performance
        rf_acc = self.performance(rf_model, X_train, X_test, y_test)
        print(f"Test RandomForest Accuracy: {rf_acc}")

        # XGBoost model performance
        xgb_acc = self.performance(xgb_model, X_train, X_test, y_test)
        print(f"Test XGBoost Accuracy: {xgb_acc}")

        # feature importance using RandomForest
        rf_feat_impt_df = self.feat_importance(rf_model, X_test)
        print(rf_feat_impt_df)
        data_path = "../data/rf_feat_impt_df.csv"
        self.save_data(rf_feat_impt_df, data_path)

        # feature importance using XGBoost
        xgb_feat_impt_df = self.feat_importance(xgb_model, X_test)
        print(xgb_feat_impt_df)
        data_path = "../data/xgb_feat_impt_df.csv"
        self.save_data(xgb_feat_impt_df, data_path)

        # correlation matrix
        self.corr_matrix(merged_df)

        return None


if __name__ == "__main__":
    sys.exit(LabelingModel().main())
