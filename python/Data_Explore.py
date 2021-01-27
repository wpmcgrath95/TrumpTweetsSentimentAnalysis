import os
import sys

import pandas as pd
from textblob import TextBlob


class SentimentOfTweets(object):
    def __init__(self):
        # load the trump tweets directly from repository
        this_dir = os.path.dirname(os.path.realpath(__file__))
        trump_data_dir = os.path.join(this_dir, "../data/realdonaldtrump.csv")
        sentiment_data_dir = os.path.join(this_dir, "../data/sentLib.csv")
        self.tweets_df = pd.read_csv(trump_data_dir, sep=",")

        print(sentiment_data_dir)

        # self.polarity = poloarity

    # load the sentiment library
    # def sentiment_library(self):
    #  sent_lib_df = pd.read_csv(SENTIMENT_URL,sep=',')

    #   # split on '#' symbol to isolate sentiment words
    #    sent_lib_df[['Lemma','PoS']] = sent_lib_df["# lemma#PoS"].str.split("#",
    #                                                            1, expand=True)
    #     sent_lib_df = sent_lib_df.drop('# lemma#PoS', 1)
    #      sent_lib_df = sent_lib_df.rename(columns={'Lemma': 'word',
    #                    'prior_polarity_score': 'polarity', 'PoS': 'pos'})
    #       sent_lib_df['word'] = sent_lib_df['word'].replace({'_': ' '}, regex=True)
    #
    #      # convert dataframe into a dictionary
    #       sent_dict = sent_lib_df.set_index('word').T.to_dict('list')
    #
    #    return sent_dict

    #    @staticmethod
    # def most_frequent(List) -> str:
    #   # finding which values in set are nans
    #    set_list = set(List)
    #     set_list = {x for x in set_list if x==x}
    #      try:
    #           max_list = max(set_list, key = List.count)
    #
    #      except ValueError:
    #           max_list = np.nan
    #
    #    return max_list

    # def to_sentiment(self, polarity):
    #   # set target variable based on average polarity score with 3 rules
    #   polarity = np.round(polarity, 2)
    #   if not math.isnan(polarity):
    #       if polarity >= -1 and polarity <= -0.05:
    #           return -1
    #       elif polarity > -0.05 and polarity <= 0.05:
    #           return 0
    #       else:
    #           return 1
    #   else:
    #       return np.nan

    def main(self):
        # pass in sentiment library
        # sent_dict = self.sentiment_library()

        # convert tweet into a list of words
        # self.tweets_df['content_list'] = self.tweets_df['content'].str.split()

        # get median polarity per list of words in each tweet
        # self.tweets_df['med_poly_scre']=self.tweets_df['content_list'].apply(lambda x:
        #                                 np.nanmean([sent_dict.get(word.lower())[0] if
        #                                 sent_dict.get(word.lower()) is not None else
        #                                 np.nan for word in x]))

        # get mode of part of speech in tweet
        # self.tweets_df['most_used_pos']=self.tweets_df['content_list'].apply(lambda x:
        #                          self.most_frequent([sent_dict.get(word.lower())[1] if
        #                          sent_dict.get(word.lower()) is not None else np.nan
        #                          for word in x]))

        # set target column
        # self.tweets_df['targ']=self.tweets_df['med_poly_scre'].apply(self.to_sentiment)

        # print(self.tweets_df)
        # print(self.tweets_df['targ'].value_counts())

        # test TextBlob
        wiki = TextBlob("test")
        print(wiki.sentiment)


if __name__ == "__main__":
    sys.exit(SentimentOfTweets().main())
