#!/usr/bin/env python3
import os
import re
import sys

import numpy as np
import pandas as pd
from wordcloud import WordCloud

np.random.seed(2)


# add LDA model to group Tweets and add it as a feature
class LDAGrouping(object):
    def __init__(self):
        # load the trump tweets directly from repository
        self.this_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(self.this_dir, "../data/realdonaldtrump.csv")
        self.tweets_df = pd.read_csv(data_dir, sep=",")

    def process_tweet(self):
        # convert to lowercase letters
        self.tweets_df["content_pro"] = self.tweets_df["content"].map(
            lambda x: x.lower()
        )

        # remove punctuation
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: re.sub(r"[,\.!?]", "", x)
        )

        # remove numbers
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: re.sub("[0-9]", "", x)
        )

        # remove URLs (treat as raw string with r)
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: re.sub(r"https?://\S+", "", x)
        )

        # remove NonASCII characters
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: re.sub(r"[^\x00-\x7F]+", " ", x)
        )

        # remove beginning, and end whitespaces
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: re.sub(r"^\s+|\s+$", "", x)
        )

        # remove duplicate whitespaces
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: " ".join(re.split(r"\s+", x))
        )

        # remove quotes
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: x.replace('"', "")
        )

        return None

    def word_cloud(self):
        # join different processed tweets together
        long_string = ",".join(list(self.tweets_df["content_pro"].values))

        # Create a WordCloud object
        word_cloud = WordCloud(
            background_color="white",
            max_words=5000,
            contour_width=3,
            contour_color="steelblue",
        )

        # Generate a word cloud
        word_cloud.generate(long_string)

        # save word cloud as a png file
        word_cloud_img = os.path.join(self.this_dir, "../data/word_cloud.png")
        word_cloud.to_file(word_cloud_img)

        return None

    def main(self):
        self.process_tweet()
        print(self.tweets_df["content_pro"])

        self.word_cloud()


if __name__ == "__main__":
    sys.exit(LDAGrouping().main())
