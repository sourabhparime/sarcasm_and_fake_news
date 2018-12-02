# -*- coding: UTF-8 -*-
import nltk
import re
import emoji
from nltk.corpus import stopwords
from nltk import word_tokenize

stops = set(stopwords.words('english'))

def replace_emoticons(tweet):
    
    tweet = emoji.demojize(tweet) 
    return tweet
    #print(tweet)

def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    tweet = re.sub('@[^\s]+',' ',tweet)
    tweet = re.sub(r'\d+','',tweet)
    tweet = re.sub('[!,?,:,.,;,/,$,%,^,*,"]+',' ',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub('&amp', ' ', tweet)
    tweet = re.sub('&gt', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    tweet = ' '.join(w.strip("'") for w in tweet.split())
    tweet = ' '.join(w.strip("-") for w in tweet.split())
    return tweet

def compress_chars(tweet):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", tweet)

def remove_stops(tweet):
    words = word_tokenize(tweet)
    for word in words:
        #word = "".join(w for w in word)
        word = ''.join([c for c in word if ord(c) < 128])

    wl = [word for word in words if word not in stops]
    return " ".join(w for w in wl)

# test 
#p = pre()
# test all funcs here
