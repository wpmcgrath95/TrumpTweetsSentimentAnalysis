import os

import pandas as pd


# add LDA model to group Tweets and add it as a feature
class LDAGrouping(object):
    def __init__(self):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(this_dir, "../data/realdonaldtrump.csv")
        self.tweets_df = pd.read_csv(data_dir, sep=",")

    def main(self):
        pass


if __name__ == "__main__":
    pass
