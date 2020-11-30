#!/usr/bin/env python3
import os
import re
import sys
import warnings

import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis
import seaborn as sns
from pyLDAvis import sklearn as sklearn_lda
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=FutureWarning)


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

        # create a WordCloud object
        word_cloud = WordCloud(
            background_color="white",
            max_words=5000,
            contour_width=3,
            contour_color="steelblue",
        )

        # generate a word cloud
        word_cloud.generate(long_string)

        # save word cloud as a png file
        word_cloud_img = os.path.join(self.this_dir, "../plots/word_cloud.png")
        word_cloud.to_file(word_cloud_img)

        return None

    def most_common_words(self, count_data, count_vectorizer):
        # converting the docs into a vector repr (bag of words)
        # then converting list of tweets into lists of vectors all with len = to voc
        all_words = count_vectorizer.get_feature_names()
        tot_counts = np.zeros(len(all_words))
        for count in count_data:
            tot_counts = tot_counts + count.toarray()[0]

        # top 10 most common words
        count_dict = zip(all_words, tot_counts)
        count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
        words = [i[0] for i in count_dict]
        counts = [i[1] for i in count_dict]
        words_range = np.arange(len(words))

        # plot 10 most common words
        plt.figure(2, figsize=(15, 10))
        plt.subplot(title="10 Most Common Words")
        sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
        sns.set_style("whitegrid")
        sns.barplot(words_range, counts, palette="husl")
        plt.xticks(words_range, words, rotation=90)
        plt.xlabel("Words")
        plt.ylabel("Counts")

        # save plot as a png file
        common_words_plt = os.path.join(self.this_dir, "../plots/common_words.png")
        plt.savefig(common_words_plt, bbox_inches="tight")

        return None

    def lda_model(self, count_data, n_topics) -> LDA:
        # create and fit the LDA model
        lda = LDA(
            n_components=n_topics,
            topic_word_prior=0.1,
            doc_topic_prior=0.1,
            n_jobs=-1,
            random_state=0,
        )
        lda_fitted = lda.fit(count_data)

        return n_topics, lda_fitted

    def main(self):
        # set seed
        np.random.seed(2)

        # processed data
        self.process_tweet()

        # word cloud
        self.word_cloud()

        # initialise the count vectorizer with English stop words
        count_vectorizer = CountVectorizer(stop_words="english")

        # fit and transform the processed tweets
        count_data = count_vectorizer.fit_transform(self.tweets_df["content_pro"])

        # most common words
        self.most_common_words(count_data, count_vectorizer)

        # fitted lda model with 20 topics
        n_topics, lda_fitted = self.lda_model(count_data, 20)

        # set LDAvis_prepared paths
        LDAvis_prep_data_path = os.path.join(
            self.this_dir, "../data/ldavis_data_" + str(n_topics)
        )
        LDAvis_prep_html_path = os.path.join(
            self.this_dir, "../plots/ldavis_html_" + str(n_topics)
        )

        # load LDAvis_prepared data from disk
        # plot showing topics in topic model that has been fitted to corpus of text data
        try:
            with open(LDAvis_prep_data_path, "rb") as f:
                LDAvis_prep = cPickle.load(f)

        except FileNotFoundError:
            LDAvis_prep = sklearn_lda.prepare(lda_fitted, count_data, count_vectorizer)
            with open(LDAvis_prep_data_path, "wb") as f:
                cPickle.dump(LDAvis_prep, f)

            # save html file
            pyLDAvis.save_html(LDAvis_prep, LDAvis_prep_html_path + ".html")

        # need to return groups as well
        return LDAvis_prep_html_path


if __name__ == "__main__":
    sys.exit(LDAGrouping().main())
