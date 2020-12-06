#!/usr/bin/env python3
# BDA 696 Final Project
# Create by: Will McGrath

"""
Input: Donald Trump Tweets

Description:
    set target variable based on average polarity score with 3 rules
    1. choose 100 random tweets and label them
    2. build a model on those 100 random tweets
    3. use this model to predict the rest of the tweets and
    4. check to see the ones that were scored well so check ones
        that scored 1 or 0 and some in the middle, and label those.
        400 or more labeled
    5. build model on prev stuff and make that ground truth for unsup model
    6. try transforming data like box-cox to improve performance
    7. then use PCA
    8. then unsupervised model (k-means) and check perf of ones you labeled

Output: Docker file that can ran with sentiement of Tweets
"""
import os
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

        # import LDA model
        (
            processed_content,
            self.LDAvis_prep_html_path,
            self.most_com_words,
        ) = LDAGrouping().main()
        self.tweets_df["processed_content"] = processed_content

    @staticmethod
    def print_heading(title: str) -> str:
        # creates headers to divide outputs
        print("\n")
        print("*" * 90)
        print(title)
        print("*" * 90)

        return None

    @staticmethod
    def get_count(most_com_words: list, tweet: str) -> int:
        overall_cnt = 0
        for word in most_com_words:
            word_cnt = TextBlob(tweet).words.count(word)
            overall_cnt = overall_cnt + word_cnt

        return overall_cnt

    def to_sentiment(self, polarity):
        # target column (determine sentiment based on other model)
        """
        polarity = np.round(polarity, 2
        if not math.isnan(polarity):
            if polarity >= -1 and polarity <= -0.33:
                return -1
            elif polarity > -0.33 and polarity <= 0.33:
                return 0
            else:
                return 1
        else:
            return np.nan
        """
        pass

    def feature_engineering(self):
        # might want some tweets like hashtags to use unprocessed content
        # feat 1: day of week (Monday, Tuesday, etc.)
        self.tweets_df["date"] = pd.to_datetime(
            self.tweets_df.date, format="%Y-%m-%d %H:%M:%S"
        )
        self.tweets_df["day_of_week"] = self.tweets_df["date"].dt.day_name()

        # feat 2: MAGA count
        self.tweets_df["MAGA_count"] = self.tweets_df["processed_content"].apply(
            lambda tweet: TextBlob(tweet).words.count("MAGA")
        )

        # feat 3: word count
        self.tweets_df["word_count"] = self.tweets_df["processed_content"].apply(
            lambda tweet: len(tweet.split())
        )

        # feat 4: character count
        self.tweets_df["character_count"] = self.tweets_df["processed_content"].apply(
            lambda tweet: len(tweet)
        )

        # feat 5: elapsed time (time since last tweet)
        position = self.tweets_df.columns.get_loc("date")
        self.tweets_df["elapsed_time"] = (
            self.tweets_df.iloc[1:, position] - self.tweets_df.iat[0, position]
        )
        elapsed_col = self.tweets_df.elapsed_time
        self.tweets_df["elapsed_time"] = elapsed_col.dt.total_seconds()

        # feat 6: get how many of the 10 most common words are in tweet
        self.tweets_df["com_word_cnt"] = self.tweets_df["processed_content"].apply(
            lambda tweet: self.get_count(self.most_com_words, tweet)
        )

        return None

    def upsample(self):
        # upsample class distribution
        pass

    def train(self):
        pass

    def pipeline(self):
        pass

    def main(self):
        # set seed
        np.random.seed(1)
        # spell check words in each tweet or row (takes a long time to run..)
        # self.tweets_df["crt_spell"]=self.tweets_df["content"].apply(lambda t:
        # TextBlob(t).correct())

        # add features to tweets_df
        self.feature_engineering()

        # subjectivity score - opinion vs non-opinion
        # i.e. 1=opinion and 0=fact
        self.tweets_df["subjectivity_score"] = self.tweets_df[
            "processed_content"
        ].apply(lambda tweet: TextBlob(tweet).sentiment[1])

        # get polarity of each tweet
        self.tweets_df["polarity_score"] = self.tweets_df["processed_content"].apply(
            lambda tweet: TextBlob(tweet).sentiment[0]
        )

        # response var
        self.tweets_df["target"] = self.tweets_df["polarity_score"].apply(
            self.to_sentiment
        )

        print(self.tweets_df.head(25))
        print(self.tweets_df["target"].value_counts())

        # opens LDAvis_prepared data
        webbrowser.open(
            "file://" + os.path.realpath(self.LDAvis_prep_html_path + ".html")
        )

        return None


if __name__ == "__main__":
    sys.exit(SentimentOfTweets().main())
