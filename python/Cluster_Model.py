#!/usr/bin/env python3
# Create by: Will McGrath

import gc
import os
import sys
from itertools import cycle

import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize


class ClusterModel(object):
    def __init__(self):
        # load the non-labeled data with feats and labeled data
        self.this_dir = os.path.dirname(os.path.realpath(__file__))

    def load_pipeline(self, path):
        # load pipeline
        model_dir = os.path.join(self.this_dir, path)
        with open(model_dir, "rb") as f:
            pipeline = cPickle.load(f)

        return pipeline

    def load_data(self, orig_row_len, orig_col_len):
        # load the merged df and merged df with nulls
        data_dir = os.path.join(self.this_dir, "../data")
        nulls_merged_data = os.path.join(
            data_dir, "merged_nulls_labeled_feat_tweets.csv"
        )
        no_nulls_merged_data = os.path.join(data_dir, "merged_labeled_feat_tweets.csv")

        # open data as dataframe
        nulls_merged_df = pd.read_csv(nulls_merged_data, sep=",")
        no_nulls_merged_df = pd.read_csv(no_nulls_merged_data, sep=",")

        # dataset shapes before concat
        null_rows = nulls_merged_df.shape[0]
        null_cols = nulls_merged_df.shape[1]
        no_null_rows = no_nulls_merged_df.shape[0]
        no_null_cols = no_nulls_merged_df.shape[1]

        # check if add rows is correct
        assert null_rows + no_null_rows == orig_row_len, "Rows Not Equal"

        # check if add cols is correct
        assert null_cols == no_null_cols, "Cols Not Equal"
        assert null_cols == orig_col_len, "Cols Not Equal"

        return no_nulls_merged_df, nulls_merged_df

    def predict_data(self, df, model):
        # create X_test
        X_test = df.iloc[:, :-1]

        # set predictions as target
        df.iloc[:, -1] = model.predict(X_test)

        return df

    def df_concat(self, df1, df2, orig_row_len, orig_col_len):
        # return total dataframe with all labels
        res = pd.concat([df1, df2])

        # check if concatented rows correctly
        assert res.shape[0] == orig_row_len, "Rows Concat Not Equal"

        # check if concatented cols correctly
        assert res.shape[1] == orig_col_len, "Cols Concat Not Equal"

        return res

    def feat_scaling(self, df):
        # train and test split
        # define X = predictors and y = response var
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        # feature scaling so zero mean and unit variance
        stand_scaler = StandardScaler()

        # fit only on training set
        stand_scaler.fit(X_train)

        # transform both train and test sets
        X_train = stand_scaler.transform(X_train)
        X_test = stand_scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def select_n_components(self, var_ratio, goal_var: float) -> int:
        # create a function to decide number of components
        # set initial variance explained so far
        tot_variance = 0.0

        # set initial number of features
        n_comps = 0

        # for the explained variance of each feat
        for exp_variance in var_ratio:

            # add the explained variance to the total
            tot_variance = tot_variance + exp_variance

            # add one to the number of components
            n_comps = n_comps + 1

            # if reach our goal level of explained variance
            if tot_variance >= goal_var:
                break

        return n_comps

    def get_max_n_comps(self, X_train, y_train):
        # create and run an LinearDiscriminantAnalysis
        linda_model = LinearDiscriminantAnalysis()
        linda_model.fit(X_train, y_train)

        # create array of explained variance ratios
        # amount of variance held after dim reduction
        linda_var_ratios = linda_model.explained_variance_ratio_

        # get num of comps
        # min num of comps such that 95% of the variance is retained
        n_comps = self.select_n_components(linda_var_ratios, 0.95)

        return n_comps

    def dim_reduction(self, X_train, y_train, X_test, y_test, model):
        if isinstance(model, LinearDiscriminantAnalysis):
            # LinearDiscriminantAnalysis since we have labeled data
            # only fit train data
            model.fit(X_train, y_train)

            # transform train and test
            X_train = model.transform(X_train)
            X_test = model.transform(X_test)

        else:
            # fit train data
            model.fit(X_train)

            # transform train and test
            X_train = model.transform(X_train)
            X_test = model.transform(X_test)

        return X_train, y_train, X_test, y_test

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

    def cluster_plot(self, X_test, y_test, model):
        fig, ax = plt.subplots(figsize=(10, 10))
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(
            np.arange(
                start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01
            ),
            np.arange(
                start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01
            ),
        )
        plt.contourf(
            X1,
            X2,
            model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75,
            cmap=ListedColormap(("red", "green", "blue")),
        )
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(
                X_set[y_set == j, 0],
                X_set[y_set == j, 1],
                c=ListedColormap(("red", "green", "blue"))(i),
                label=j,
            )
        plt.title("Random Forest (Test set)")
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        plt.legend()

        # save plot
        plt_name = f"../plots/rf_LinDA_cluster_.png"
        linda_cluster_plt = os.path.join(self.this_dir, plt_name)
        fig.savefig(linda_cluster_plt, bbox_inches="tight")

        return None

    def perf(self, name, pred, y_test):
        pca_performance_cols = ["Precision", "Recall", "F1"]
        pca_performance_df = pd.DataFrame(
            columns=pca_performance_cols,
        )

        # for label inbalance
        prec = precision_score(y_test, pred, average="weighted")
        recall = recall_score(y_test, pred, average="weighted")
        f1 = f1_score(y_test, pred, average="weighted")

        pca_performance_df.loc[name] = pd.Series(
            {
                "Precision": prec,
                "Recall": recall,
                "F1": f1,
            }
        )

        perf_dir = os.path.join(self.this_dir, "../plots/pca_perf.csv")
        pca_performance_df.to_csv(perf_dir, encoding="utf-8", index=False)

        return None

    def main(self):
        # set seed
        np.random.seed(1)

        # shape of original CSV with feats (CSV_with_feats cols - 5)
        orig_row_len = 43352
        orig_col_len = 100
        diff_2 = 289
        diff_1 = 120
        diff_0 = 32

        # load model
        xgb_pipeline = self.load_pipeline("../models/xgb_pipeline.pickle")

        # load datasets
        no_nulls_merged_df, nulls_merged_df = self.load_data(orig_row_len, orig_col_len)

        # predict target on nulls
        nulls_merged_df = self.predict_data(nulls_merged_df, xgb_pipeline["model"])

        # nulls_merged_df targ distr
        null_targ_distr = nulls_merged_df["target"].value_counts()

        # concat both dfs so labeled one and predicted one
        all_merged_df = self.df_concat(
            no_nulls_merged_df, nulls_merged_df, orig_row_len, orig_col_len
        )

        # all_merged_df targ distr
        all_targ_distr = all_merged_df["target"].value_counts()

        # assert concat makes sense
        assert all_targ_distr[2] - null_targ_distr[2] == diff_2, "Targ 2 Not Correct"
        assert all_targ_distr[1] - null_targ_distr[1] == diff_1, "Targ 1 Not Correct"
        assert all_targ_distr[0] - null_targ_distr[0] == diff_0, "Targ 0 Not Correct"
        assert all_merged_df["target"].isna().sum() == 0, "N/A Vals in Target"

        # print final concat dataset shape and targ distr
        print(
            f"Dataset:[{all_merged_df.shape[0]} rows x {all_merged_df.shape[1]} cols]"
        )
        print(all_targ_distr)

        # feature scaling - standardize
        X_train, X_test, y_train, y_test = self.feat_scaling(all_merged_df)

        # get max n-components for LinearDiscriminantAnalysis model
        n_comps = self.get_max_n_comps(X_train, y_train)

        # LinearDiscriminantAnalysis model
        linda_model = LinearDiscriminantAnalysis(n_components=n_comps)
        X_train, y_train, X_test, y_test = self.dim_reduction(
            X_train, y_train, X_test, y_test, linda_model
        )

        # try using Random Forest
        rf_model = RandomForestClassifier(random_state=0)

        # fit RF model
        rf_model.fit(X_train, y_train)

        # RF model predictions
        rf_y_hat = rf_model.predict(X_test)

        confusion_matric = confusion_matrix(y_test, rf_y_hat)
        print(f"Confusion Matrix LinDA: {confusion_matric}")

        # print(rf_model.feature_importances_)

        rf_linda_cr = classification_report(y_test, rf_y_hat)
        print(f"RandomForest LinDA Class Report: {rf_linda_cr}")

        # ROC Curve
        self.roc_curve(X_test, y_test, rf_model, "rf_LinDA")

        self.cluster_plot(X_test, y_test, rf_model)

        # remove temp dataframes from memory
        gc.collect()

        # PCA - feature scaling - standardize
        X_train, X_test, y_train, y_test = self.feat_scaling(all_merged_df)

        # make PCA model (0.95 variability)
        pca_model = PCA(0.95)

        # get scaled data
        X_train, y_train, X_test, y_test = self.dim_reduction(
            X_train, y_train, X_test, y_test, pca_model
        )

        # RF model PCA
        rf_model_pca = RandomForestClassifier(random_state=2)

        # fit RF model PCA
        rf_model_pca.fit(X_train, y_train)

        # RF model predictions
        rf_y_hat_pca = rf_model_pca.predict(X_test)

        confusion_matric = confusion_matrix(y_test, rf_y_hat_pca)
        print(f"Confusion Matrix PCA: {confusion_matric}")

        rf_pca_cr = classification_report(y_test, rf_y_hat_pca)
        print(f"RandomForest PCA Class Report: {rf_pca_cr}")

        # ROC Curve
        self.roc_curve(X_test, y_test, rf_model_pca, "rf_PCA")

        # 37 feats left
        var_df = pd.DataFrame(
            pca_model.explained_variance_ratio_, columns=["Var of Prin. Comp."]
        )
        var_df["Cum. Sum"] = var_df["Var of Prin. Comp."].cumsum()
        var_df = var_df[var_df["Cum. Sum"] < 0.95]
        print(var_df)

        self.perf("RandomForest_PCA", rf_y_hat_pca, y_test)

        return None


if __name__ == "__main__":
    sys.exit(ClusterModel().main())
