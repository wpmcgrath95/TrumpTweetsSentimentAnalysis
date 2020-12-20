import os
import sys

import _pickle as cPickle
import numpy as np
import pandas as pd
from fastai.text.all import AWD_LSTM, TextDataLoaders, language_model_learner


# deep learning using fastai
class TrumpifyTweet(object):
    def __init__(self):
        self.this_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(self.this_dir, "../data/twitter_data_with_feats.csv")
        self.tweets_df = pd.read_csv(data_dir, sep=",")

    def main(self, input_tweet, num_add_words):
        # set seed
        np.random.seed(1)

        # getting DataLoader
        dls = TextDataLoaders.from_df(
            self.tweets_df,
            path="../data/",
            text_col="processed_content",
            seed=3,
            valid_pct=0.3,
            is_lm=True,
        )

        # train model
        model_dir = os.path.join(self.this_dir, "../models/lstm_model.pickle")
        try:
            with open(model_dir, "rb") as f:
                lstm_model = cPickle.load(f)

        except FileNotFoundError:
            with open(model_dir, "wb") as f:
                # LSTM model to generate sentences
                lstm_model = language_model_learner(dls, AWD_LSTM)
                lstm_model.fit_one_cycle(5)
                cPickle.dump(lstm_model, f)

        # predict sentence
        pred_output = lstm_model.predict(input_tweet, num_add_words)

        return pred_output


if __name__ == "__main__":
    input_tweet = input("Please enter a Tweet: ")
    num_add_words = int(
        input("Please enter the number of Trump words you would like to add: ")
    )
    sys.exit(TrumpifyTweet().main(input_tweet, num_add_words))
