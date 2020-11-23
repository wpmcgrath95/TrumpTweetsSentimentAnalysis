import sys
import pandas as pd
import math
import numpy as np
import os
from textblob import TextBlob

class SentimentOfTweets(object):
    def __init__(self):
        # load the trump tweets directly from repository
        this_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(this_dir, "../data/realdonaldtrump.csv")
        self.tweets_df = pd.read_csv(data_dir,sep=',')
    
    def to_sentiment(self, polarity):
        # set target variable based on average polarity score with 3 rules
        polarity = np.round(polarity, 2)
        if not math.isnan(polarity):
            if polarity >= -1 and polarity <= -0.34:
                return -1
            elif polarity > -0.33 and polarity <= 0.33:
                return 0
            else:
                return 1
        else:
            return np.nan

    def feature_engineering(self):
        # feature 1: subjectivity (opinion vs non-opinion, 1=opinion and 0=fact)
        self.tweets_df['subjectivity_score'] = self.tweets_df['content'].apply(
            lambda tweet: TextBlob(tweet).sentiment[1])
        # feature 2: day of the week (Monday, Tuesday, etc.) 
        self.tweets_df['date'] = pd.to_datetime(self.tweets_df.date, format='%Y-%m-%d %H:%M:%S')
        self.tweets_df['day_of_week'] = self.tweets_df['date'].dt.day_name()

        return None

    def upsample(self):
        # upsample class distribution
        pass

    def main(self):
        # spell check words (takes a long time to run..)
        # self.tweets_df['content'] = self.tweets_df['content'].apply(lambda tweet: TextBlob(tweet).correct())

        # get polarity of each tweet
        self.tweets_df['polarity_score'] = self.tweets_df['content'].apply(lambda tweet:TextBlob(tweet).sentiment[0])

        # set target column
        self.tweets_df['target'] = self.tweets_df['polarity_score'].apply(self.to_sentiment)
        
        print(self.tweets_df.head(25))
        print(self.tweets_df['target'].value_counts())

if __name__ == "__main__":
    sys.exit(SentimentOfTweets().main())