#!/usr/bin/env python3
# BDA 696 Final Project
# Create by: Will McGrath

"""
Prototype model

Input: Donald Trump Tweets up to June 2020

Description:
    1. choose 100 random tweets and label them (label Tweets that are new and some old)
    2. build a model on those 100 random tweets
    3. use this model to predict the rest of the tweets and
    4. check to see the ones that were scored well so check ones
        that scored 1 or 0 and some in the middle, and label those.
        400 or more labeled
    5. build model on prev stuff and make that ground truth for unsup model
    6. transforming data like box-cox, and scale data (standardize) to improve perf
        try checking for importance, and multicorrelation (use VIF)
    7. then use PCA
    8. then unsupervised model (k-means) and check perf of ones you labeled or
        can try SVM (use bi-grams)
    9. plot mean of response, try brute force and correlation plots to
       get rid of unnecessary feats

Plots:
     - plot polarity vs subj distr

Output: Sentiment predictions on Trump Tweets
        - used Docker file
        - add a volumn to Docker and output all plots and stuff to folder in vol
        Note: the sentiment being predicted is how Trump feels about the subject
              in the tweet
"""
import gc
import os
import re
import sys
import webbrowser

import numpy as np
import pandas as pd
from LDA_Grouping import LDAGrouping
from sklearn.preprocessing import OneHotEncoder
from textblob import TextBlob

pd.set_option("display.max_columns", None)


class SentimentOfTweets(object):
    def __init__(self):
        # load the trump tweets directly from repository
        self.this_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(self.this_dir, "../data/realdonaldtrump.csv")
        self.tweets_df = pd.read_csv(data_dir, sep=",")

        # import Trump's processed content w/ 10 most common words and topics
        (
            processed_content,
            self.LDAvis_prep_html_path,
            self.most_comm_words,
            self.unique_topics,
        ) = LDAGrouping().main()
        self.tweets_df["processed_content"] = processed_content["content_pro"]

        # feat: predicted topic of tweet
        self.tweets_df["topic"] = processed_content["topic"]

    @staticmethod
    def print_heading(title: str) -> str:
        # creates headers to divide outputs
        print("\n")
        print("*" * 90)
        print(title)
        print("*" * 90)

        return None

    @staticmethod
    def comm_word_count(most_comm_words: list, istweet: bool, tweet: str) -> int:
        overall_cnt = 0
        if istweet:
            for word in most_comm_words:
                word_cnt = TextBlob(tweet).words.count(word)
                overall_cnt = overall_cnt + word_cnt
        elif not istweet and type(tweet) == str:
            for word in most_comm_words:
                word_cnt = len(re.findall(word, tweet.lower()))
                overall_cnt = overall_cnt + word_cnt
        else:
            pass

        return overall_cnt

    def feature_engineering(self):
        # some tweets like hashtags use unprocessed content
        self.tweets_df["date"] = pd.to_datetime(
            self.tweets_df.date, format="%Y-%m-%d %H:%M:%S"
        )

        # shifting dates to get rid of hours
        self.tweets_df["no_hr_date"] = pd.to_datetime(
            self.tweets_df["date"]
        ).dt.strftime("%m-%d-%Y")

        self.tweets_df["no_hr_date"] = pd.to_datetime(
            self.tweets_df["no_hr_date"], format="%m-%d-%Y"
        )

        # feat: day of week (Monday, Tuesday, etc.)
        self.tweets_df["day_of_week"] = self.tweets_df["no_hr_date"].dt.day_name()

        # feat: number of tweets on day or per row (always 1)
        self.tweets_df["num_of_tweets"] = 1

        # feat: day in last 7 days with most tweets (most active day)

        # feat: most common day in last 10 tweets

        # feat: number of unique days tweeted in last 7 days

        # feat: number of unique days tweeted in last 30 days

        # feat: MAGA count
        self.tweets_df["MAGA_count"] = self.tweets_df["processed_content"].apply(
            lambda tweet: TextBlob(tweet).words.count("MAGA")
        )

        # feat: word count
        self.tweets_df["word_count"] = self.tweets_df["processed_content"].apply(
            lambda tweet: len(tweet.split())
        )

        # feat: character count
        self.tweets_df["character_count"] = self.tweets_df["processed_content"].apply(
            lambda tweet: len(tweet)
        )

        # feat: number of hrs from previous tweet
        self.tweets_df["time_diff_hrs"] = (
            self.tweets_df["date"]
            .diff()
            .apply(lambda x: np.round(x.total_seconds() / 3600), 2)
            .fillna(0)
        )

        # feat: number of days from previous tweet
        self.tweets_df["time_diff_days"] = (
            self.tweets_df["no_hr_date"].diff().apply(lambda x: x.days).fillna(0)
        )

        # feat: number of days in last 10 tweets
        self.tweets_df["num_days_last_10_rows"] = (
            self.tweets_df["time_diff_days"].rolling(window=10).sum()
        )

        # feat: mean number of days in last 10 tweets
        self.tweets_df["mean_days_last_10_rows"] = (
            self.tweets_df["time_diff_days"].rolling(window=10).mean()
        )

        # feat: number of days in last 30 tweets
        self.tweets_df["num_days_last_30_rows"] = (
            self.tweets_df["time_diff_days"].rolling(window=30).sum()
        )

        # feat: mean number of days in last 30 tweets
        self.tweets_df["mean_days_last_30_rows"] = (
            self.tweets_df["time_diff_days"].rolling(window=30).mean()
        )

        # feat: number of days in last 90 tweets
        self.tweets_df["num_days_last_90_rows"] = (
            self.tweets_df["time_diff_days"].rolling(window=90).sum()
        )

        # feat: mean number of days in last 90 tweets
        self.tweets_df["mean_days_last_90_rows"] = (
            self.tweets_df["time_diff_days"].rolling(window=90).mean()
        )

        # feat: number of days in last 180 tweets
        self.tweets_df["num_days_last_180_rows"] = (
            self.tweets_df["time_diff_days"].rolling(window=180).sum()
        )

        # feat: mean number of days in last 180 tweets
        self.tweets_df["mean_days_last_180_rows"] = (
            self.tweets_df["time_diff_days"].rolling(window=180).mean()
        )

        # feat: get how many of the 10 most common words are in a tweet
        self.tweets_df["tweet_comm_word_cnt"] = self.tweets_df[
            "processed_content"
        ].apply(lambda tweet: self.comm_word_count(self.most_comm_words, True, tweet))

        # feat: number of tweets in last 3 days (maybe add .fillna(0))
        num_tw_3_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=3)
            .sum()["num_of_tweets"]
        )

        # feat: mean number of tweets in last 3 days
        mean_tw_3_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=3)
            .mean()["num_of_tweets"]
        )
        temp_df = pd.merge(num_tw_3_df, mean_tw_3_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "num_of_tweets_x": "num_tweets_last_3_days",
                "num_of_tweets_y": "mean_tweets_last_3_days",
            },
            inplace=True,
        )

        # feat: number of tweets in last 7 days
        num_tw_7_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=7)
            .sum()["num_of_tweets"]
        )
        temp_df = pd.merge(temp_df, num_tw_7_df, how="inner", on=["no_hr_date"])

        # feat: mean number of tweets in last 7 days
        mean_tw_7_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=7)
            .mean()["num_of_tweets"]
        )
        temp_df = pd.merge(temp_df, mean_tw_7_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "num_of_tweets_x": "num_tweets_last_7_days",
                "num_of_tweets_y": "mean_tweets_last_7_days",
            },
            inplace=True,
        )

        # feat: number of tweets in last 14 days
        num_tw_14_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=14)
            .sum()["num_of_tweets"]
        )
        temp_df = pd.merge(temp_df, num_tw_14_df, how="inner", on=["no_hr_date"])

        # feat: mean number of tweets in last 14 days
        mean_tw_14_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=14)
            .mean()["num_of_tweets"]
        )
        temp_df = pd.merge(temp_df, mean_tw_14_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "num_of_tweets_x": "num_tweets_last_14_days",
                "num_of_tweets_y": "mean_tweets_last_14_days",
            },
            inplace=True,
        )

        # feat: number of tweets in last 50 days
        num_tw_50_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=50)
            .sum()["num_of_tweets"]
        )
        temp_df = pd.merge(temp_df, num_tw_50_df, how="inner", on=["no_hr_date"])

        # feat: mean number of tweets in last 50 days
        mean_tw_50_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=50)
            .mean()["num_of_tweets"]
        )
        temp_df = pd.merge(temp_df, mean_tw_50_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "num_of_tweets_x": "num_tweets_last_50_days",
                "num_of_tweets_y": "mean_tweets_last_50_days",
            },
            inplace=True,
        )

        # merge with main twitter dataframe
        self.tweets_df = pd.merge(
            self.tweets_df, temp_df, how="inner", on=["no_hr_date"]
        )

        # delete all temp dfs
        del [
            [
                num_tw_3_df,
                mean_tw_3_df,
                num_tw_7_df,
                mean_tw_7_df,
                num_tw_14_df,
                mean_tw_14_df,
                num_tw_50_df,
                mean_tw_50_df,
                temp_df,
            ]
        ]

        # feat: number of hastags in a tweet
        self.tweets_df["num_of_hashtags"] = self.tweets_df["hashtags"].apply(
            lambda tweet: len(tweet.split(",")) if type(tweet) == str else 0
        )

        # feat: how many of the 10 most common words are in tweet's hashtag(s)
        self.tweets_df["hashtag_comm_word_cnt"] = self.tweets_df["hashtags"].apply(
            lambda tweet: self.comm_word_count(self.most_comm_words, False, tweet)
        )

        # feat: is topic in tweet's hashtag(s)

        # feat: check if any of the topics are in tweet's hashtag(s)

        # feat: mean hashtag count in last 3 days
        mean_ht_3_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=3)
            .mean()["num_of_hashtags"]
        )

        # feat: mean hashtag count in last 7 days
        mean_ht_7_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=7)
            .mean()["num_of_hashtags"]
        )
        temp_df = pd.merge(mean_ht_3_df, mean_ht_7_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "num_of_hashtags_x": "mean_hashtag_last_3_days",
                "num_of_hashtags_y": "mean_hashtag_last_7_days",
            },
            inplace=True,
        )

        # feat: mean hashtag count in last 14 days
        mean_ht_14_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=14)
            .mean()["num_of_hashtags"]
        )
        temp_df = pd.merge(temp_df, mean_ht_14_df, how="inner", on=["no_hr_date"])

        # feat: mean hashtag count in last 50 days
        mean_ht_50_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=50)
            .mean()["num_of_hashtags"]
        )
        temp_df = pd.merge(temp_df, mean_ht_50_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "num_of_hashtags_x": "mean_hashtag_last_14_days",
                "num_of_hashtags_y": "mean_hashtag_last_50_days",
            },
            inplace=True,
        )

        # feat: most used hashtag in last 3 days

        # feat: most used hashtag in last 7 days

        # feat: most used hashtag in last 14 days

        # feat: most used hashtag in last 50 days

        # merge with main twitter dataframe
        self.tweets_df = pd.merge(
            self.tweets_df, temp_df, how="inner", on=["no_hr_date"]
        )

        # delete all temp dfs
        del [[mean_ht_3_df, mean_ht_7_df, mean_ht_14_df, mean_ht_50_df, temp_df]]

        # feat: mean hashtag count in last 3 tweets
        self.tweets_df["mean_hashtag_last_3_rows"] = (
            self.tweets_df["num_of_hashtags"].rolling(window=3).mean()
        )

        # feat: mean hashtag count in last 5 tweets
        self.tweets_df["mean_hashtag_last_5_rows"] = (
            self.tweets_df["num_of_hashtags"].rolling(window=5).mean()
        )

        # feat: mean hashtag count in last 10 tweets
        self.tweets_df["mean_hashtag_last_10_rows"] = (
            self.tweets_df["num_of_hashtags"].rolling(window=10).mean()
        )

        # feat: mean hashtag count in last 50 tweets
        self.tweets_df["mean_hashtag_last_50_rows"] = (
            self.tweets_df["num_of_hashtags"].rolling(window=50).mean()
        )

        # feat: most used hashtag in last 3 tweets

        # feat: most used hashtag in last 5 tweets

        # feat: most used hashtag in last 10 tweets

        # feat: most used hashtag in last 50 tweets

        # feat: unique hashtag count in last 7 days

        # feat: unique hashtag count in last 3 tweets

        # feat: number of mentions in a tweet
        self.tweets_df["num_of_mentions"] = self.tweets_df["mentions"].apply(
            lambda tweet: len(tweet.split(",")) if type(tweet) == str else 0
        )

        # feat: how many of the 10 most common words are in a tweet's mention(s)
        self.tweets_df["mention_comm_word_cnt"] = self.tweets_df["mentions"].apply(
            lambda tweet: self.comm_word_count(self.most_comm_words, False, tweet)
        )

        # feat: mean mention count in last 3 days
        mean_mt_3_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=3)
            .mean()["num_of_mentions"]
        )

        # feat: mean mention count in last 7 days
        mean_mt_7_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=7)
            .mean()["num_of_mentions"]
        )
        temp_df = pd.merge(mean_mt_3_df, mean_mt_7_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "num_of_mentions_x": "mean_mention_last_3_days",
                "num_of_mentions_y": "mean_mention_last_7_days",
            },
            inplace=True,
        )

        # feat: mean mention count in last 14 days
        mean_mt_14_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=14)
            .mean()["num_of_mentions"]
        )
        temp_df = pd.merge(temp_df, mean_mt_14_df, how="inner", on=["no_hr_date"])

        # feat: mean mention count in last 50 days
        mean_mt_50_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=50)
            .mean()["num_of_mentions"]
        )
        temp_df = pd.merge(temp_df, mean_mt_50_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "num_of_mentions_x": "mean_mention_last_14_days",
                "num_of_mentions_y": "mean_mention_last_50_days",
            },
            inplace=True,
        )

        # feat: most used mention in last 3 days

        # feat: most used mention in last 7 days

        # feat: most used mention in last 14 days

        # feat: most used mention in last 50 days

        # feat: mean mention count in last 3 tweets
        self.tweets_df["mean_mention_last_3_rows"] = (
            self.tweets_df["num_of_mentions"].rolling(window=3).mean()
        )

        # feat: mean mention count in last 5 tweets
        self.tweets_df["mean_mention_last_5_rows"] = (
            self.tweets_df["num_of_mentions"].rolling(window=5).mean()
        )

        # feat: mean mention count in last 10 tweets
        self.tweets_df["mean_mention_last_10_rows"] = (
            self.tweets_df["num_of_mentions"].rolling(window=10).mean()
        )

        # feat: mean mention count in last 50 tweets
        self.tweets_df["mean_mention_last_50_rows"] = (
            self.tweets_df["num_of_mentions"].rolling(window=50).mean()
        )

        # feat: most used mention in last 3 tweets

        # feat: most used mention in last 5 tweets

        # feat: unique mention count in last 7 days

        # feat: unique mention count in last 3 tweets

        # merge with main twitter dataframe
        self.tweets_df = pd.merge(
            self.tweets_df, temp_df, how="inner", on=["no_hr_date"]
        )

        # delete all temp dfs
        del [[mean_mt_3_df, mean_mt_7_df, mean_mt_14_df, mean_mt_50_df, temp_df]]

        # feat: diff b/t number of retweets from previous tweet
        self.tweets_df["RT_diff"] = self.tweets_df["retweets"].diff()

        # feat: mean number of retweets in last 3 days
        mean_rt_3_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=3)
            .mean()["retweets"]
        )

        # feat: mean number of retweets in last 7 days
        mean_rt_7_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=7)
            .mean()["retweets"]
        )
        temp_df = pd.merge(mean_rt_3_df, mean_rt_7_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "retweets_x": "mean_retweet_last_3_days",
                "retweets_y": "mean_retweet_last_7_days",
            },
            inplace=True,
        )

        # feat: mean number of retweets in last 14 days
        mean_rt_14_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=14)
            .mean()["retweets"]
        )
        temp_df = pd.merge(temp_df, mean_rt_14_df, how="inner", on=["no_hr_date"])

        # feat: mean number of retweets in last 50 days
        mean_rt_50_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=50)
            .mean()["retweets"]
        )
        temp_df = pd.merge(temp_df, mean_rt_50_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "retweets_x": "mean_retweet_last_14_days",
                "retweets_y": "mean_retweet_last_50_days",
            },
            inplace=True,
        )

        # feat: mean number of retweets in last 3 tweets
        self.tweets_df["mean_retweet_last_3_rows"] = (
            self.tweets_df["retweets"].rolling(window=3).mean()
        )

        # feat: mean number of retweets in last 5 tweets
        self.tweets_df["mean_retweet_last_5_rows"] = (
            self.tweets_df["retweets"].rolling(window=5).mean()
        )

        # feat: mean number of retweets in last 10 tweets
        self.tweets_df["mean_retweet_last_10_rows"] = (
            self.tweets_df["retweets"].rolling(window=10).mean()
        )

        # feat: mean number of retweets in last 50 tweets
        self.tweets_df["mean_retweet_last_50_rows"] = (
            self.tweets_df["retweets"].rolling(window=50).mean()
        )

        # merge with main twitter dataframe
        self.tweets_df = pd.merge(
            self.tweets_df, temp_df, how="inner", on=["no_hr_date"]
        )

        # delete all temp dfs
        del [[mean_rt_3_df, mean_rt_7_df, mean_rt_14_df, mean_rt_50_df, temp_df]]

        # feat: diff b/t number of favorites from previous tweet
        self.tweets_df["fav_diff"] = self.tweets_df["favorites"].diff()

        # feat: mean number of favorites in last 3 days
        mean_fav_3_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=3)
            .mean()["favorites"]
        )

        # feat: mean number of favorites in last 7 days
        mean_fav_7_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=7)
            .mean()["favorites"]
        )
        temp_df = pd.merge(mean_fav_3_df, mean_fav_7_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "favorites_x": "mean_fav_last_3_days",
                "favorites_y": "mean_fav_last_7_days",
            },
            inplace=True,
        )

        # feat: mean number of favorites in last 14 days
        mean_fav_14_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=14)
            .mean()["favorites"]
        )
        temp_df = pd.merge(temp_df, mean_fav_14_df, how="inner", on=["no_hr_date"])

        # feat: mean number of favorites in last 50 days
        mean_fav_50_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=50)
            .mean()["favorites"]
        )
        temp_df = pd.merge(temp_df, mean_fav_50_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "favorites_x": "mean_fav_last_14_days",
                "favorites_y": "mean_fav_last_50_days",
            },
            inplace=True,
        )

        # feat: mean number of favorites in last 3 tweets
        self.tweets_df["mean_fav_last_3_rows"] = (
            self.tweets_df["favorites"].rolling(window=3).mean()
        )

        # feat: mean number of favorites in last 5 tweets
        self.tweets_df["mean_fav_last_5_rows"] = (
            self.tweets_df["favorites"].rolling(window=5).mean()
        )

        # feat: mean number of favorites in last 10 tweets
        self.tweets_df["mean_fav_last_10_rows"] = (
            self.tweets_df["favorites"].rolling(window=10).mean()
        )

        # feat: mean number of favorites in last 50 tweets
        self.tweets_df["mean_fav_last_50_rows"] = (
            self.tweets_df["favorites"].rolling(window=50).mean()
        )

        # merge with main twitter dataframe
        self.tweets_df = pd.merge(
            self.tweets_df, temp_df, how="inner", on=["no_hr_date"]
        )

        # delete all temp dfs
        del [[mean_fav_3_df, mean_fav_7_df, mean_fav_14_df, mean_fav_50_df, temp_df]]

        # feat: number of unique topic in last 7 days

        # feat: most used topic in last 3 days

        # feat: most used topic in last 7 days

        # feat: most used topic in last 14 days

        # feat: most used topic in last 50 days

        # feat: most used topic last 3 tweets

        # feat: most used topic last 5 tweets

        # feat: most used topic last 10 tweets

        # feat: most used topic last 50 tweets

        # feat: mean topic last 3 days

        # feat: mean topic last 7 days

        # feat: mean topic last 14 days

        # feat: mean topic last 50 days

        # feat: mean topic last 3 tweets
        # self.tweets_df["mean_topic_last_3_rows"] = (
        #    self.tweets_df["topic"].rolling(window=3).mean()
        # )

        # feat: mean topic last 5 tweets

        # feat: mean topic last 10 tweets

        # feat: mean topic last 50 tweets

        # feat: tweet's subjectivity
        # subjectivity score: opinion vs fact (score is a number between 0.0 and 1.0)
        # i.e. 0 = very objective (fact-based), 1.0 = very subjective (opinion-based)
        self.tweets_df["subj_score"] = self.tweets_df["processed_content"].apply(
            lambda tweet: TextBlob(tweet).sentiment[1]
        )

        # feat: diff in subjectivity from previous tweet
        self.tweets_df["subj_diff"] = self.tweets_df["subj_score"].diff()

        # feat: mean subjectivity in last 3 tweets
        self.tweets_df["mean_subj_score_last_3_rows"] = (
            self.tweets_df["subj_diff"].rolling(window=3).mean()
        )

        # feat: mean subjectivity in last 5 tweets
        self.tweets_df["mean_subj_score_last_5_rows"] = (
            self.tweets_df["subj_diff"].rolling(window=5).mean()
        )

        # feat: mean subjectivity in last 10 tweets
        self.tweets_df["mean_subj_score_last_10_rows"] = (
            self.tweets_df["subj_diff"].rolling(window=10).mean()
        )

        # feat: mean subjectivity in last 50 tweets
        self.tweets_df["mean_subj_score_last_50_rows"] = (
            self.tweets_df["subj_diff"].rolling(window=50).mean()
        )

        # feat: mean subjectivity in all tweets from previous day

        # feat: mean subjectivity in last 3 days
        mean_subj_3_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=3)
            .mean()["subj_score"]
        )

        # feat: mean subjectivity in last 7 days
        mean_subj_7_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=7)
            .mean()["subj_score"]
        )
        temp_df = pd.merge(
            mean_subj_3_df, mean_subj_7_df, how="inner", on=["no_hr_date"]
        )
        temp_df.rename(
            columns={
                "subj_score_x": "mean_subj_last_3_days",
                "subj_score_y": "mean_subj_last_7_days",
            },
            inplace=True,
        )

        # feat: mean subjectivity in last 14 days
        mean_subj_14_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=14)
            .mean()["subj_score"]
        )
        temp_df = pd.merge(temp_df, mean_subj_14_df, how="inner", on=["no_hr_date"])

        # feat: mean subjectivity in last 50 days
        mean_subj_50_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=50)
            .mean()["subj_score"]
        )
        temp_df = pd.merge(temp_df, mean_subj_50_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "subj_score_x": "mean_subj_last_14_days",
                "subj_score_y": "mean_subj_last_50_days",
            },
            inplace=True,
        )

        # feat: highest subjectivity in last 3 tweets

        # feat: highest subjectivity in last 5 tweets

        # feat: highest subjectivity in last 10 tweets

        # feat: highest subjectivity in last 50 tweets

        # feat: highest subjectivity in last 3 days

        # merge with main twitter dataframe
        self.tweets_df = pd.merge(
            self.tweets_df, temp_df, how="inner", on=["no_hr_date"]
        )

        # delete all temp dfs
        del [
            [mean_subj_3_df, mean_subj_7_df, mean_subj_14_df, mean_subj_50_df, temp_df]
        ]

        # feat: tweet's polarity (if correlation with target need 3)
        # polarity score: score is a number between -1.0 and 1.0
        # i.e. -1.0 = very negative, 0 = neutral, and 1 = very positive
        self.tweets_df["poly_score"] = self.tweets_df["processed_content"].apply(
            lambda tweet: TextBlob(tweet).sentiment[0]
        )

        # feat: diff in polarity from previous tweet
        self.tweets_df["poly_diff"] = self.tweets_df["poly_score"].diff()

        # feat: mean polarity in last 3 tweets
        self.tweets_df["mean_poly_score_last_3_rows"] = (
            self.tweets_df["poly_diff"].rolling(window=3).mean()
        )

        # feat: mean polarity in last 5 tweets
        self.tweets_df["mean_poly_score_last_5_rows"] = (
            self.tweets_df["poly_diff"].rolling(window=5).mean()
        )

        # feat: mean polarity in last 10 tweets
        self.tweets_df["mean_poly_score_last_10_rows"] = (
            self.tweets_df["poly_diff"].rolling(window=10).mean()
        )

        # feat: mean polarity in last 50 tweets
        self.tweets_df["mean_poly_score_last_50_rows"] = (
            self.tweets_df["poly_diff"].rolling(window=50).mean()
        )

        # feat: mean polarity in all tweets from previous day

        # feat: mean polarity in last 3 day
        mean_poly_3_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=3)
            .mean()["poly_score"]
        )

        # feat: mean polarity in last 7 day
        mean_poly_7_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=7)
            .mean()["poly_score"]
        )
        temp_df = pd.merge(
            mean_poly_3_df, mean_poly_7_df, how="inner", on=["no_hr_date"]
        )
        temp_df.rename(
            columns={
                "poly_score_x": "mean_poly_last_3_days",
                "poly_score_y": "mean_poly_last_7_days",
            },
            inplace=True,
        )

        # feat: mean polarity in last 14 day
        mean_poly_14_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=14)
            .mean()["poly_score"]
        )
        temp_df = pd.merge(temp_df, mean_poly_14_df, how="inner", on=["no_hr_date"])

        # feat: mean polarity in last 50 day
        mean_poly_50_df = (
            self.tweets_df.resample("D", on="no_hr_date")
            .sum()
            .rolling(window=50)
            .mean()["poly_score"]
        )
        temp_df = pd.merge(temp_df, mean_poly_50_df, how="inner", on=["no_hr_date"])
        temp_df.rename(
            columns={
                "poly_score_x": "mean_poly_last_14_days",
                "poly_score_y": "mean_poly_last_50_days",
            },
            inplace=True,
        )

        # feat: highest polarity in last 3 tweets

        # feat: highest polarity in last 5 tweets

        # feat: highest polarity in last 10 tweets

        # feat: highest polarity in last 50 tweets

        # feat: highest polarity in last 3 days

        # merge with main twitter dataframe
        self.tweets_df = pd.merge(
            self.tweets_df, temp_df, how="inner", on=["no_hr_date"]
        )

        # delete all temp dfs
        del [
            [mean_poly_3_df, mean_poly_7_df, mean_poly_14_df, mean_poly_50_df, temp_df]
        ]

        # feat: most common words in negative tweets

        # feat: most common words in neutral tweets

        # feat: most common words in positive tweets

        # feat: num of words in tweet that are “pos” keywords

        # feat: num of words in tweet that are “neg” keywords

        # feat: num of words in tweet that are “neu” keywords

        # feat: common verbs in tweet

        # feat: amount of misspellings

        # feat: part of speech in tweet

        # feat: verbs or nouns in tweet

        return None

    def preprocess_feats(self):
        # clean and preprocess rest of feats (tweets are processed)
        # need to drop mentions and hashtags feats before imputation
        self.tweets_df.drop(["mentions", "hashtags"], axis=1, inplace=True)

        # set index to date
        self.tweets_df.set_index("no_hr_date", inplace=True)

        # imputation (i.e. replacing missing values in usable feats)
        # might want to just use .fillna(method='bfill')
        # since time-series using observed vals after date
        self.tweets_df = self.tweets_df.interpolate(method="time").bfill()

        # reset index
        self.tweets_df.reset_index(inplace=True)

        # check if feats have any more missing values
        na_list = self.tweets_df.columns[self.tweets_df.isna().any()].tolist()
        assert len(na_list) == 0, "List is not empty"

        return None

    def encode(self):
        # transform categorical feats
        categ_cols = ["topic", "day_of_week"]
        enc_df = pd.DataFrame(
            OneHotEncoder().fit_transform(self.tweets_df[categ_cols]).toarray()
        )
        self.tweets_df = self.tweets_df.join(enc_df)

        return None

    def main(self):
        # set seed
        np.random.seed(1)

        # add features to tweets_df
        self.feature_engineering()

        # remove temp dataframes from memory
        gc.collect()

        # preprocessing feats (i.e. deal with missing data)
        self.preprocess_feats()

        # one hot encoding categorical cols
        self.encode()

        # final dataset shape before training
        print(
            f"Dataset:[{self.tweets_df.shape[0]} rows x {self.tweets_df.shape[1]} cols]"
        )

        # opens LDAvis_prepared data
        webbrowser.open(
            "file://" + os.path.realpath(self.LDAvis_prep_html_path + ".html")
        )

        # save twitter df with data and feats to csv in data folder
        feat_dir = os.path.join(self.this_dir, "../data/twitter_data_with_feats.csv")
        self.tweets_df.to_csv(feat_dir, encoding="utf-8", index=False)

        return None


if __name__ == "__main__":
    sys.exit(SentimentOfTweets().main())
