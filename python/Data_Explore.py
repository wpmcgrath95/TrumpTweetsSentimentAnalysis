# Import Libraries
import sys
import pandas as pd
import numpy as np

def main():

    # Load the trump tweets directly from repository
    url = 'https://github.com/wpmcgrath95/TrumpTweetsSentimentAnalysis/blob/data_and_feat_engineering/data/realdonaldtrump.csv?raw=true'
    fdt = pd.read_csv(url,sep=',')

    # Load the sentiment library
    url2 = 'https://github.com/wpmcgrath95/TrumpTweetsSentimentAnalysis/blob/data_and_feat_engineering/data/sentLib.csv?raw=true'
    sentlib = pd.read_csv(url2,sep=',')  

    # Split on '#' symbol to isolate sentiment words
    sentlib[['Lemma','PoS']] = sentlib["# lemma#PoS"].str.split("#", 1, expand=True)
 
    # Convert sentiment library to dictionary 
    sent = dict(zip(sentlib.Lemma, sentlib.prior_polarity_score))

    # Convert to lowercase
    fdt2 = fdt['content'].str.lower()

    # Convert to seperate words
    fdt_split = fdt2.str.split(expand=True)

    # Calculate average polarity for each tweet
    polarity = np.zeros(shape=(43352,1))
    for row in fdt_split:
        for word in fdt_split[row]:
            if word in sent:
                word = sent[word]
                polarity[row] += word
            else:
                word = 0.0
                polarity[row] += word
        fdt['polarity'] = polarity[:]
    print(fdt.head())

if __name__ == "__main__":
    sys.exit(main())