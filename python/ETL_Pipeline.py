#!/usr/bin/env python3
# BDA 696 Final Project
# Create by: Will McGrath

"""
Pipeline for model (ETL process)

Input: Donald Trump Tweets up to June 2020

Output: Sentiment predictions on Trump Tweets
"""
import os
import sys

import _pickle as cPickle
from sklearn.pipeline import Pipeline


class ETLPipeline(object):
    def __init__(self):
        self.this_dir = os.path.dirname(os.path.realpath(__file__))

    def pre_process(self):
        # scale data using column transformer
        # combine all transformers (add custom transf)
        pass

    def main(self):
        # add gridsearch to maximize recall

        # set pipeline pickle file path
        pipeline_dir = os.path.join(self.this_dir, "../models/pipeline.pickle")

        # save and load trained etl_pipeline as a pickle file
        try:
            with open(pipeline_dir, "rb") as f:
                etl_pipeline = cPickle.load(f)

        except FileNotFoundError:
            with open(pipeline_dir, "wb") as f:
                # fit and predict
                etl_pipeline = Pipeline(steps=[])
                cPickle.dump(etl_pipeline, f)

        # print performance results
        predictions = etl_pipeline.predict()

        return predictions


if __name__ == "__main__":
    sys.exit(ETLPipeline().main())
