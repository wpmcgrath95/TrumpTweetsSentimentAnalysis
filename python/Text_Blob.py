#!/usr/bin/env python3
# BDA 696 Final Project
# Create by: Will McGrath

"""
Input: Donald Trump tweets up to June 2020

Description:
    1. choose 100 random tweets and label them (label Tweetws that are new and some old)
    2. build a model on those 100 random tweets
    3. use this model to predict the rest of the tweets and
    4. check to see the ones that were scored well so check ones
        that scored 1 or 0 and some in the middle, and label those.
        400 or more labeled
    5. build model on prev stuff and make that ground truth for unsup model
    6. transforming data like box-cox, and scale data (standardize) to improve perf
    7. then use PCA
    8. then unsupervised model (k-means) and check perf of ones you labeled

Output: Docker file that can be ran to predict the sentiment behind Trump tweets
        Note: the sentiment being predicted is how Trump feels about the subject
              in the tweet
"""
import os
import re
import sys
import webbrowser

import numpy as np
import pandas as pd
from LDA_Grouping import LDAGrouping
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
        # feat: day of week (Monday, Tuesday, etc.)
        self.tweets_df["date"] = pd.to_datetime(
            self.tweets_df.date, format="%Y-%m-%d %H:%M:%S"
        )
        self.tweets_df["day_of_week"] = self.tweets_df["date"].dt.day_name()

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

        # feat: elapsed time (time since last tweet)
        position = self.tweets_df.columns.get_loc("date")
        self.tweets_df["last_tweet_elapsed_time"] = (
            self.tweets_df.iloc[1:, position] - self.tweets_df.iat[0, position]
        )
        elapsed_col = self.tweets_df.last_tweet_elapsed_time
        self.tweets_df["last_tweet_elapsed_time"] = elapsed_col.dt.total_seconds()

        # feat: number of days (24 hrs) from previous tweet
        self.tweets_df["time_diff"] = (
            self.tweets_df["date"].diff().apply(lambda x: x.days).fillna(0)
        )

        # feat: number of days in last 90 tweets
        self.tweets_df["days_last_90_rows"] = (
            self.tweets_df["time_diff"].rolling(window=90).sum()
        )

        # feat: number of days in last 180 tweets
        self.tweets_df["days_last_180_rows"] = (
            self.tweets_df["time_diff"].rolling(window=180).sum()
        )

        # feat: get how many of the 10 most common words are in a tweet
        self.tweets_df["tweet_comm_word_cnt"] = self.tweets_df[
            "processed_content"
        ].apply(lambda tweet: self.comm_word_count(self.most_comm_words, True, tweet))

        # feat: number of hastags
        self.tweets_df["hashtag_cnt"] = self.tweets_df["hashtags"].apply(
            lambda tweet: len(tweet.split(",")) if type(tweet) == str else 0
        )

        # feat: how many of the 10 most common words are in tweet's hashtag(s)
        self.tweets_df["hashtag_comm_word_cnt"] = self.tweets_df["hashtags"].apply(
            lambda tweet: self.comm_word_count(self.most_comm_words, False, tweet)
        )

        # feat: is topic in tweet's hashtag(s)

        # feat: check if any of the topics are in tweet's hashtag(s)

        # feat: most used hashtag in last 7 days

        # feat: number of mentions
        self.tweets_df["mention_cnt"] = self.tweets_df["mentions"].apply(
            lambda tweet: len(tweet.split(",")) if type(tweet) == str else 0
        )

        # feat: how many of the 10 most common words are in a tweet's mention(s)
        self.tweets_df["mention_comm_word_cnt"] = self.tweets_df["mentions"].apply(
            lambda tweet: self.comm_word_count(self.most_comm_words, False, tweet)
        )

        # feat: number of tweets in last 7 days

        # feat: mean number of tweets in last 7 days

        # feat: diff b/t number of retweets from previous tweet
        self.tweets_df["RT_diff"] = self.tweets_df["retweets"].diff()

        # feat: number of retweets in last 7 days

        # feat: diff b/t number of favorites from previous tweet
        self.tweets_df["fav_diff"] = self.tweets_df["favorites"].diff()

        # feat: number of favorites in last 7 days

        # feat: most used topic in last 3 days

        # feat: most used topic in last 7 days

        # feat: mean topic last 7 days

        # feat: tweet's subjectivity
        # subjectivity score: opinion vs fact (score is a number between 0.0 and 1.0)
        # i.e. 0 = very objective (fact-based), 1.0 = very subjective (opinion-based)
        self.tweets_df["subj_score"] = self.tweets_df["processed_content"].apply(
            lambda tweet: TextBlob(tweet).sentiment[1]
        )

        # feat: diff in subjectivity from previous tweet
        self.tweets_df["subj_diff"] = self.tweets_df["subj_score"].diff()

        # feat: mean subjectivity in last 3 day

        # feat: mean subjectivity in last 3 tweets
        self.tweets_df["mean_subj_score_last_3_rows"] = (
            self.tweets_df["subj_diff"].rolling(window=3).mean()
        )

        # feat: mean subjectivity in all tweets from previous day

        # feat: mean subjectivity in last 7 day

        # feat: tweet's polarity
        # polarity score: score is a number between -1.0 and 1.0
        # i.e. 1.0 = very negative, 0 = neutral, and 1 = very positive
        self.tweets_df["polarity_score"] = self.tweets_df["processed_content"].apply(
            lambda tweet: TextBlob(tweet).sentiment[0]
        )

        # feat: diff in polarity from previous tweet
        self.tweets_df["polarity_diff"] = self.tweets_df["polarity_score"].diff()

        # feat: mean polarity in last 3 day

        # feat: mean polarity in last 3 tweets
        self.tweets_df["mean_polarity_score_last_3_rows"] = (
            self.tweets_df["polarity_diff"].rolling(window=3).mean()
        )

        # feat: mean polarity in all tweets from previous day

        # feat: mean polarity in last 7 day

        # feat: most common words in negative tweets

        # feat: most common words in neutral tweets

        # feat: most common words in positive tweets

        # feat: common verbs in tweet

        # feat: amount of misspellings

        # feat: part of speech in tweet

        return None

    def upsample(self):
        # upsample class distribution
        pass

    def train(self):
        pass

    def pipeline(self):
        # add gridsearch to maximize recall
        pass

    def performance(self):
        # ROC curve, PPV, TPR, F1, and SHAP values
        pass

    def main(self):
        # set seed
        np.random.seed(1)

        # add features to tweets_df
        self.feature_engineering()

        # need to add to target
        # response variable (need to predict) and count
        # self.tweets_df["target"] = ""
        # print(f"Target Value Count: {self.tweets_df['target'].value_counts()}")
        print(
            f"Dataset: [{self.tweets_df.shape[0]} rows x {self.tweets_df.shape[1]} cols]"
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
