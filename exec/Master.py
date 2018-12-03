import pandas as pd 
import numpy as np
import codecs
import pre, NB
from sklearn.model_selection import train_test_split

# Read the file and create a dict of tweets
def read(path = '../data/Annnotated_Copy.xlsx'):
    df = pd.read_excel(path, 'Annotated')
    tweets = df[['tweet_id', 'text', 'is_sarcasm']]
    return tweets

def clean_frame(tweets):
    
    tweets['text'] = tweets['text'].apply(pre.replace_emoticons)
    tweets['text'] = tweets['text'].apply(pre.clean_tweet)
    tweets['text'] = tweets['text'].apply(pre.compress_chars)
    #tweets['text'] = tweets['text'].apply(pre.remove_stops)
    return tweets

cleaned_df = clean_frame(read())
# save the cleaned df
cleaned_df.to_csv("cleaned.csv")
X = cleaned_df['text']
y = cleaned_df['is_sarcasm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
 random_state=42)

train_data = pd.concat([X_train, y_train], axis=1).reset_index()


sar_tweets = []
nonosar_tweets = []
for index, row in train_data.iterrows():
    if row['is_sarcasm'] == True:
        sar_tweets.append((row['text'], 'sarcastic'))
    else:
        sar_tweets.append((row['text'], 'not_sarcastic'))
# call naive bayes model and print classification report
print(NB.get_accuracy(sar_tweets,nonosar_tweets, X_test ))
 


