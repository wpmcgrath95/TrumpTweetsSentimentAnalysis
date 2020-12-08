import unittest


class LDAUnitTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

        """
        test:
        # fitted lda model with 20 topics
        n_topics, lda_model_fitted = self.lda_model(count_data, 20)

        # lda_model_fitted performance
        log_like_fitted, perp_fitted = self.performance(count_data,lda_model_fitted)
        print("Model: lda_model_fitted",end="\n")
        print(f"Log Likelihood: {log_like_fitted}")
        print(f"Perplexity: {perp_fitted}")

        is better than best lda_model_fitted
        """


if __name__ == "__main__":
    unittest.main()
