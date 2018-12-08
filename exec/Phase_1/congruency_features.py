import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer







def get_text_length(x):
    return np.array([len(t) for t in x]).reshape(-1, 1)

def get_pos_words(x):
    sid = SentimentIntensityAnalyzer()
    pos = []
    for s in x:
        count = 0    
        for word in list(s):
            if sid.polarity_scores(word)['compound'] >=0.5: count += 1
        pos.append(count)
    return np.array(pos).reshape(-1,1)

def get_neg_words(x):
    sid = SentimentIntensityAnalyzer()
    pos = []
    for s in x:
        count = 0    
        for word in list(s):
            if sid.polarity_scores(word)['compound'] <=-0.5: count += 1
        pos.append(count)
    return np.array(pos).reshape(-1,1)

def get_compound(x):
    sid = SentimentIntensityAnalyzer()
    pos = []
    for s in x:
        pos.append(sid.polarity_scores(s)['compound'])
    return np.array(pos).reshape(-1,1)

def get_number_of_flips(x):
    sid = SentimentIntensityAnalyzer()
    flips = []
    for s in x:
        count = 0
        words = list(s)
        for i, word in enumerate(words[1:]):
            if abs(sid.polarity_scores(word)['compound'] - sid.polarity_scores(words[i-1])['compound']) > 0.5: count += 1
        flips.append(count)
    return np.array(flips).reshape(-1,1)

def get_sentiment_flip(x):
    sid = SentimentIntensityAnalyzer()
    flips = []
    for s in x:
        count = 0
        m = 0
        words = list(s)
        for i, word in enumerate(words[1:]):
            if abs(sid.polarity_scores(word)['compound'] - sid.polarity_scores(words[i-1])['compound']) <= 0.5: count += 1
            else:
                m = max(count, m)
                count = 0
        flips.append(m)
    return np.array(flips).reshape(-1,1)

def get_compound_mnb(x):
    #
    """
    Formula to scale 
    ((maxnew−minnew)/(maxold−minold))⋅(v−maxold)+maxnew
    """
    sid = SentimentIntensityAnalyzer()
    pos = []
    for s in x:
        raw = sid.polarity_scores(s)['compound']
        scaled = ((1)/(2)) * (raw - 1) + 1
        pos.append(scaled)
    return np.array(pos).reshape(-1,1)




