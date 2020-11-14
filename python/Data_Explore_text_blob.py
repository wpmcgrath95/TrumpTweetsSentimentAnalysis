import sys
import pandas as pd
import math
import numpy as np
from textblob import TextBlob

TWEETS_URL = 'https://github.com/wpmcgrath95/TrumpTweetsSentimentAnalysis/blob/data_and_feat_engineering/data/realdonaldtrump.csv?raw=true'

class SentimentOfTweets(object):
    def __init__(self):
        # load the trump tweets directly from repository
        self.tweets_df = pd.read_csv(TWEETS_URL,sep=',')
    
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

    def main(self):
        # spell check words (takes a long time to run..)
        #self.tweets_df['content'] = self.tweets_df['content'].apply(lambda tweet: TextBlob(tweet).correct())

        # get polarity of each tweet
        self.tweets_df['polarity_score'] = self.tweets_df['content'].apply(lambda tweet: TextBlob(tweet).sentiment[0])

        # set target column
        self.tweets_df['target'] = self.tweets_df['polarity_score'].apply(self.to_sentiment)
        
        print(self.tweets_df.head(25))
        print(self.tweets_df['target'].value_counts())

if __name__ == "__main__":
    sys.exit(SentimentOfTweets().main())