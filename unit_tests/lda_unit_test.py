#!/usr/bin/env python3
# BDA 696 Final Project
# Create by: Will McGrath
import unittest


class LDAUnitTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

        """
        test best_lda_model is better than:
        # fitted lda model with 20 topics
        n_topics, lda_model_fitted = self.lda_model(count_data, 20)

        # lda_model_fitted performance
        log_like_fitted, perp_fitted = self.performance(count_data,lda_model_fitted)
        print("Model: lda_model_fitted",end="\n")
        print(f"Log Likelihood: {log_like_fitted}")
        print(f"Perplexity: {perp_fitted}")

        log_like_best, perp_best = self.performance(count_data,best_lda_model)
        print("Model: best_lda_model",end="\n")
        print(f"Best Model's Params: {best_lda_model.get_params()}")
        print(f"Log Likelihood: {log_like_best}")
        print(f"Perplexity: {perp_best}")
        """


if __name__ == "__main__":
    unittest.main()
