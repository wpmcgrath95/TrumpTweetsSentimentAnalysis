import sys
import pandas as pd
from textblob import TextBlob

TWEETS_URL = 'https://github.com/wpmcgrath95/TrumpTweetsSentimentAnalysis/blob/data_and_feat_engineering/data/realdonaldtrump.csv?raw=true'

class SentimentOfTweets(object):
    def __init__(self):
        # load the trump tweets directly from repository
        self.tweets_df = pd.read_csv(TWEETS_URL,sep=',')
    
    def main(self):
        # get polarity of each tweet
        self.tweets_df['polarity_score'] = self.tweets_df['content'].apply(lambda tweet: TextBlob(tweet).sentiment[0])
        print(self.tweets_df.head(50))

if __name__ == "__main__":
    sys.exit(SentimentOfTweets().main())