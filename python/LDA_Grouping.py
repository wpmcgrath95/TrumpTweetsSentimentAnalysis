#!/usr/bin/env python3
# Create by: Will McGrath

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
from sklearn.model_selection import GridSearchCV
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=FutureWarning)


# add unsupervided LDA model to group tweets and add it as a feat
class LDAGrouping(object):
    def __init__(self):
        # load the trump tweets directly from repository
        self.this_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(self.this_dir, "../data/realdonaldtrump.csv")
        self.tweets_df = pd.read_csv(data_dir, sep=",")

    def preprocess_tweets(self):
        # clean and preprocess tweet content
        # convert to lowercase letters
        self.tweets_df["content_pro"] = self.tweets_df["content"].map(
            lambda x: x.lower()
        )

        # remove punctuation
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: re.sub(r"[,\.!?]", "", x)
        )

        # remove more punctuation
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: re.sub(r"[^\w\s]", "", x)
        )

        # remove numbers
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: re.sub("[0-9]", "", x)
        )

        # remove http/https URLs (treat as raw string with r)
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: re.sub(r"https?://\S+|http[s]?\S+", "", x)
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

        # remove @ symbols
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: re.sub(r"[@]\w+", "user", x)
        )

        # remove # symbols
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: re.sub(r"#", "", x)
        )

        # remove & symbols
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: re.sub(r"&", "and", x)
        )

        # remove % symbols
        self.tweets_df["content_pro"] = self.tweets_df["content_pro"].map(
            lambda x: re.sub(r"%", "percent", x)
        )

        return None

    def word_cloud(self, img_path):
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
        word_cloud.to_file(img_path)

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

        return words

    def lda_model(self, count_data, n_topics) -> LDA:
        # create/load and fit the LDA model
        # bigger α (doc_topic_prior) = more similarity b/t topics in diff docs
        # bigger β (topic_word_prior) = more similarity b/t words in diff topics
        model_dir = os.path.join(self.this_dir, "../models/lda_model.pickle")
        try:
            with open(model_dir, "rb") as f:
                lda_model = cPickle.load(f)

        except FileNotFoundError:
            with open(model_dir, "wb") as f:
                lda_model = LDA(
                    n_components=n_topics,
                    topic_word_prior=0.1,
                    doc_topic_prior=0.1,
                    n_jobs=-1,
                    random_state=0,
                )

                lda_model.fit(count_data)
                cPickle.dump(lda_model, f)

        return n_topics, lda_model

    def grid_search(self, count_data):
        # find best LDA model (hyperparameter optimization)
        # learning decay = controls learning rate
        # n components (most important) = number of topics
        # learning_method = online (faster than batch for large datasets)
        # n jobs = 1 (means 100% of 1 CPU is used)
        model_dir = os.path.join(self.this_dir, "../models/best_lda_model.pickle")
        try:
            with open(model_dir, "rb") as f:
                best_lda_model = cPickle.load(f)

        except FileNotFoundError:
            with open(model_dir, "wb") as f:
                # search parameters to iterate through
                search_params = {
                    "n_components": [5, 10, 15, 20],  # TRY [2,3,4,5,10]
                    "learning_decay": [0.5, 0.7, 0.9],
                }

                # LDA model
                lda_model = LDA(
                    max_iter=5,
                    learning_method="online",
                    n_jobs=1,
                    random_state=0,
                )

                # use grid search to get best LDA estimator
                clf = GridSearchCV(lda_model, param_grid=search_params)
                clf.fit(count_data)
                best_lda_model = clf.best_estimator_
                cPickle.dump(best_lda_model, f)

        return best_lda_model

    def extract_topics(self, count_data, count_vectorizer, model):
        # all tweets with topics matrix
        doc_topic_matrix = model.transform(count_data)

        # column and index names
        topic_names = ["topic_" + str(i) for i in range(model.n_components)]
        doc_names = ["doc_" + str(i) for i in range(len(self.tweets_df))]

        # tweets and topics dataframe
        doc_topic_df = pd.DataFrame(
            np.round(doc_topic_matrix, 2), columns=topic_names, index=doc_names
        )

        # get dominant topic for each tweets
        dominant_topic = np.argmax(doc_topic_df.values, axis=1)
        doc_topic_df["dominant_topic"] = dominant_topic

        # top 15 keywords for each topic
        num_words = 15
        keywords = np.array(count_vectorizer.get_feature_names())
        top_keywords = [
            keywords.take((-weights).argsort()[:num_words])
            for weights in model.components_
        ]

        # topics and keywords dataframe with top 15 keywords
        topic_keywords_df = pd.DataFrame(top_keywords)
        topic_keywords_df.columns = [
            "word_" + str(i) for i in range(topic_keywords_df.shape[1])
        ]
        topic_keywords_df.index = [
            "topic_" + str(i) for i in range(topic_keywords_df.shape[0])
        ]

        # print(topic_keywords_df) to see words in each topic
        # infer topics (subjective)
        topics = [
            "Fake News/Things Trump Wants to Change as President",
            "News/Fox News/CNN",
            "America/Campaign",
            "MAGA/Golf",
            "Trump/Great/President",
        ]

        # add topics to dataframe
        topic_keywords_df["topic"] = topics

        # match topic to tweets and topics dataframe
        doc_topic_df["topic"] = doc_topic_df["dominant_topic"].apply(
            lambda t: topics[t]
        )

        # print(topic_keywords_df)
        # print(doc_topic_df)

        return topics, doc_topic_df

    def performance(self, count_data, model):
        # performance of the vectorized processed tweets (count_data)
        # log likelihood: higher the better
        log_likelihood = model.score(count_data)

        # perplexity: Lower the better
        # perplexity = exp(-1. * log-likelihood per word)
        perplexity = model.perplexity(count_data)

        return log_likelihood, perplexity

    def main(self):
        # set seed
        np.random.seed(1)

        # preprocess tweets
        self.preprocess_tweets()

        # word cloud
        word_cloud_img = os.path.join(self.this_dir, "../plots/word_cloud.png")
        if not os.path.isfile(word_cloud_img):
            self.word_cloud(word_cloud_img)

        # initialise the count vectorizer with English stop words
        count_vectorizer = CountVectorizer(stop_words="english")

        # fit and transform preprocessed tweets (counts the num of each word in vector)
        count_data = count_vectorizer.fit_transform(self.tweets_df["content_pro"])

        # most common words
        most_comm_words = self.most_common_words(count_data, count_vectorizer)

        # best fitted LDA model and num of topics
        best_lda_model = self.grid_search(count_data)
        n_topics = best_lda_model.n_components

        # best fitted LDA model performance
        log_like_best, perp_best = self.performance(count_data, best_lda_model)
        print("Model: best_lda_model", end="\n")
        print(f"Best Model's Params: {best_lda_model.get_params()}")
        print(f"Log Likelihood: {log_like_best}")
        print(f"Perplexity: {perp_best}")

        # extract topics from top keywords in each tweet
        topics, doc_topic_df = self.extract_topics(
            count_data, count_vectorizer, best_lda_model
        )

        # add topics to self.tweets_df
        self.tweets_df["topic"] = doc_topic_df["topic"].tolist()

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
            LDAvis_prep = sklearn_lda.prepare(
                best_lda_model, count_data, count_vectorizer
            )
            with open(LDAvis_prep_data_path, "wb") as f:
                cPickle.dump(LDAvis_prep, f)

            # save html file
            pyLDAvis.save_html(LDAvis_prep, LDAvis_prep_html_path + ".html")

        # returns interactive plot, groups, and 10 most common words
        return (
            self.tweets_df[["content_pro", "topic"]],
            LDAvis_prep_html_path,
            most_comm_words,
            topics,
        )


if __name__ == "__main__":
    sys.exit(LDAGrouping().main())
