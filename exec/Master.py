import pandas as pd 
import numpy as np
import codecs
import pre 


# Read the file and create a dict of tweets
def read(path = '../data/Ano.xlsx'):
    df = pd.read_excel(path, 'ElectionDay DATA')
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