# Import Libraries
import pandas as pd

# Load the dataset directly from repository
url = 'https://github.com/wpmcgrath95/TrumpTweetsSentimentAnalysis/blob/data_and_feat_engineering/data/realdonaldtrump.csv?raw=true'
fdt = pd.read_csv(url,sep=',')
print(fdt.head())

# Load the sentiment library
url2 = 'https://github.com/wpmcgrath95/TrumpTweetsSentimentAnalysis/blob/data_and_feat_engineering/data/SentiWords_1.1.txt?raw=true'
sentlib = pd.read_csv(url2,sep=',')  
print(sentlib.head())