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
import warnings
from itertools import cycle

import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore", category=UserWarning)


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

        # encode laebels (0 = -1, 1 = 1, 2 = 1)
        label_encoder = preprocessing.LabelEncoder()

        # transform labels and replace
        merged_df["target"] = label_encoder.fit_transform(merged_df["target"])

        return merged_df

    def save_data(self, df, path):
        # save data to csv in a folder
        df_dir = os.path.join(self.this_dir, path)
        df.to_csv(df_dir, encoding="utf-8", index=False)

        return None

    def save_load_pipeline(self, X_train, y_train, model, path):
        # save model
        model_dir = os.path.join(self.this_dir, path)
        try:
            with open(model_dir, "rb") as f:
                pipeline = cPickle.load(f)

        except FileNotFoundError:
            with open(model_dir, "wb") as f:
                pipeline = self.over_sampling_smote(X_train, y_train, model)
                cPickle.dump(pipeline, f)

        return pipeline

    def train_test_split(self, df):
        # train and test split
        # define X = predictors and y = response var
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        return X, y, X_train, X_test, y_train, y_test

    def baseline_model(self):
        # create dummy model to pred sentiment to compare real model to
        # don't need
        pass

    def feat_importance(self, model, X_test):
        # feature importance
        feat_impt = {}  # dict to hold feature_name: feature_importance
        fig, ax = plt.subplots(figsize=(10, 10))
        for feat, impt in zip(X_test, model.feature_importances_):
            feat_impt[feat] = impt

        if isinstance(model, RandomForestClassifier):
            feat_impt_df = pd.DataFrame.from_dict(feat_impt, orient="index").rename(
                columns={0: "Gini-importance"}
            )

            # plot RandomForest feature importance and save
            feat_impt_df = feat_impt_df["Gini-importance"].sort_values(ascending=False)
            feat_impt_df[0:15].plot(kind="bar", rot=90, ax=plt.gca())
            plt.title("RandomForest Feature Importance")
            plt.xlabel("Top 15 Features")
            plt.ylabel("Ranked Gini-importance")

            rf_feat_impt_plt = os.path.join(self.this_dir, "../plots/rf_feat_impt.png")
            fig.savefig(rf_feat_impt_plt, bbox_inches="tight")

        else:
            feat_impt_df = pd.DataFrame.from_dict(feat_impt, orient="index").rename(
                columns={0: "Gain"}
            )

            # plot XGBoost feature importance and save
            feat_impt_df = feat_impt_df["Gain"].sort_values(ascending=False)
            feat_impt_df[0:15].plot(kind="bar", rot=90, ax=plt.gca())
            plt.title("XGBoost Feature Importance")
            plt.xlabel("Top 15 Features")
            plt.ylabel("Ranked Average Gain")

            xgb_feat_impt_plt = os.path.join(
                self.this_dir, "../plots/xgb_feat_impt.png"
            )
            fig.savefig(xgb_feat_impt_plt, bbox_inches="tight")

        return None

    def shap_plot(self, model, X_test, name):
        fig, ax = plt.subplots(figsize=(10, 10))

        # another feature importance plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False)

        plt_name = f"../plots/{name}_shap_plot.png"
        shap_plt = os.path.join(self.this_dir, plt_name)
        fig.savefig(shap_plt, bbox_inches="tight")

        return None

    def permutation_plots(self, X_test, y_test, model, name):
        # randomly shuffle feats and computes change in model's perf
        # feats that impact the perf the most are the most import one
        fig, ax = plt.subplots(figsize=(10, 10))
        perm_impt = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42
        )

        # rank
        sorted_idx = perm_impt.importances_mean.argsort()[0:15]

        # box plot
        ax.boxplot(
            perm_impt.importances[sorted_idx].T,
            vert=False,
            labels=X_test.columns[sorted_idx],
        )
        ax.set_title("Top 15 Permutation Importances")

        plt_name = f"../plots/{name}_perm_impt.png"
        perm_impt_plt = os.path.join(self.this_dir, plt_name)
        fig.savefig(perm_impt_plt, bbox_inches="tight")

        return None

    def roc_curve(self, X_test, y_test, model, name):
        # predict probability and labelize
        y_score = model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]

        # compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="Micro-average ROC curve (area = {0:0.2f})"
            "".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="Macro-average ROC curve (area = {0:0.2f})"
            "".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label="ROC curve of class {0} (area = {1:0.2f})"
                "".format(i, roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")

        # save plot
        plt_name = f"../plots/{name}_roc_curve.png"
        roc_curve_plt = os.path.join(self.this_dir, plt_name)
        plt.savefig(roc_curve_plt, bbox_inches="tight")

        return None

    def corr_matrix(self, df):
        # only shows correlation between numerical features
        # test multicorrelation using VIF
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

    def over_sampling_smote(self, X_train, y_train, model):
        # oversampling class distribution
        # transform the dataset
        # before SMOTE: 2 = 289, 1 = 120, 0 = 32
        over = SMOTE(sampling_strategy={0: 60}, random_state=0)
        under = RandomUnderSampler(sampling_strategy={2: 120})

        steps = [("o", over), ("u", under), ("model", model)]
        pipeline = Pipeline(steps=steps)

        # fit dataset
        pipeline.fit(X_train, y_train)

        # fit dataset
        # smote_X, smote_y = pipeline.fit_resample(X, y)
        # print(f"Target w/SMOTE Value Count: \n {smote_y.value_counts()}")

        return pipeline

    def evaluate_model(self, X, y, pipeline, model, name):
        # define evaluation procedure
        # 20% of the data is used for testing
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

        # evaluate model
        # default roc is macro
        scoring = {
            "acc": "accuracy",
            "prec_macro": "precision_macro",
            "rec_macro": "recall_macro",
            "f1_macro": "f1_macro",
            "roc_auc_ovr": "roc_auc_ovr",
        }

        scores = cross_validate(
            pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=False
        )
        scores_no_smote = cross_validate(
            model, X, y, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=False
        )

        # summarize performance
        print(f"{name} w/SMOTE Accuracy Length: {len(scores['test_acc'])}")
        print(f"{name} w/SMOTE Accuracy: {np.mean(scores['test_acc'])}")
        print(f"{name} w/SMOTE STD: {np.std(scores['test_acc'])}")
        print(f"{name} w/SMOTE Precision-Macro: {np.mean(scores['test_prec_macro'])}")
        print(f"{name} w/SMOTE Recall-Macro: {np.mean(scores['test_rec_macro'])}")
        print(f"{name} w/SMOTE f1-Macro: {np.mean(scores['test_f1_macro'])}")
        print(f"{name} w/SMOTE ROC-Macro: {np.mean(scores['test_roc_auc_ovr'])}")
        print(f"{name} w/SMOTE ROC-Macro: {scores['test_roc_auc_ovr']}")

        print(f"{name} Accuracy Length: {len(scores_no_smote['test_acc'])}")
        print(f"{name} Accuracy: {np.mean(scores_no_smote['test_acc'])}")
        print(f"{name} STD: {np.std(scores_no_smote['test_acc'])}")
        print(f"{name} Precision-Macro: {np.mean(scores_no_smote['test_prec_macro'])}")
        print(f"{name} Recall-Macro: {np.mean(scores_no_smote['test_rec_macro'])}")
        print(f"{name} f1-Macro: {np.mean(scores_no_smote['test_f1_macro'])}")
        print(f"{name} ROC-Macro: {np.mean(scores_no_smote['test_roc_auc_ovr'])}")

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
        X, y, X_train, X_test, y_train, y_test = self.train_test_split(merged_df)

        # create RandomForest and XGBoost models
        rf_model = RandomForestClassifier(random_state=42)
        xgb_model = xgb.XGBClassifier(seed=42)

        # get fitted model in pipeline with SMOTE added
        xgb_pipe_path = "../models/xgb_pipeline.pickle"
        xgb_pipeline = self.save_load_pipeline(
            X_train, y_train, xgb_model, xgb_pipe_path
        )

        # get fitted model in pipeline with SMOTE added
        rf_pipe_path = "../models/rf_pipeline.pickle"
        rf_pipeline = self.save_load_pipeline(X_train, y_train, rf_model, rf_pipe_path)

        # xgb classification report
        xgb_y_hat = xgb_pipeline.predict(X_test)
        print("XGBoost Classification Report")
        print(classification_report(y_test, xgb_y_hat))

        # rf classification report
        rf_y_hat = rf_pipeline.predict(X_test)
        print("RandomForest Classification Report")
        print(classification_report(y_test, rf_y_hat))

        # feature importance using RandomForest
        self.feat_importance(rf_pipeline["model"], X_test)

        # feature importance using XGBoost
        self.feat_importance(xgb_pipeline["model"], X_test)

        # shap plots
        self.shap_plot(rf_pipeline["model"], X_test, "rf")
        self.shap_plot(xgb_pipeline["model"], X_test, "xgb")

        # permutation importance plots
        self.permutation_plots(X_test, y_test, rf_pipeline["model"], "rf")
        self.permutation_plots(X_test, y_test, xgb_pipeline["model"], "xgb")

        # ROC curves
        self.roc_curve(X_test, y_test, xgb_pipeline, "xgb")
        self.roc_curve(X_test, y_test, rf_pipeline, "rf")

        # correlation matrix
        self.corr_matrix(merged_df)

        # evaluate xgb model
        self.evaluate_model(X, y, xgb_pipeline, xgb_model, "XGBoost")

        # evaluate rf model
        self.evaluate_model(X, y, rf_pipeline, rf_model, "RandomForest")

        return None


if __name__ == "__main__":
    sys.exit(LabelingModel().main())
