import pandas as pd 
import numpy as np
import codecs
import pre, NB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

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
ros = RandomOverSampler()
X_r, y_r  = ros.fit_resample(np.array(X.values).reshape(-1, 1),np.array(y.values).reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(X_r,y_r, test_size=0.33,
 random_state=42)

train_data = pd.concat([pd.DataFrame(X_train, columns=['text']),
pd.DataFrame(y_train, columns=['is_sarcasm'])], axis=1).reset_index()
#print(train_data, X_test)


sar_tweets = []
nonosar_tweets = []
for index, row in train_data.iterrows():
    if row['is_sarcasm'] == True:
        sar_tweets.append((row['text'], 'sarcastic'))
    else:
        sar_tweets.append((row['text'], 'not_sarcastic'))
# call naive bayes model and print classification report
y_pred = NB.get_accuracy(sar_tweets,nonosar_tweets, X_test )
#print(y_pred, y_test)
y_true = []
y_preds = []
for val in y_test:
    if val == True: y_true.append(1)
    else: y_true.append(0)
for val in y_pred:
    if val == True: y_preds.append(1)
    else: y_preds.append(0)
print(classification_report(y_true, y_preds))
 


