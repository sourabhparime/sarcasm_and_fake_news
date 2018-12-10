import pandas as pd 
import numpy as np
import codecs
import pre, NB, SVM, congru
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from imblearn.over_sampling import RandomOverSampler
import utils_1 as utils

# Read the file and create a dict of tweets
def read(path = '../data/Annnotated_Copy.xlsx'):
    df = pd.read_excel(path, 'Annotated')
    tweets = df[['tweet_id', 'text', 'is_sarcasm','is_fake_news']]
    return tweets

def clean_frame(tweets):
    
    tweets['text'] = tweets['text'].apply(pre.replace_emoticons)
    tweets['text'] = tweets['text'].apply(pre.clean_tweet)
    tweets['text'] = tweets['text'].apply(pre.compress_chars)
    #tweets['text'] = tweets['text'].apply(pre.remove_stops)
    return tweets

def random_over_sample(df):
    X = df[['text','is_fake_news']]
    y = df['is_sarcasm']
    print(X.shape, y.shape)
    ros = RandomOverSampler()
    X_r, y_r  = ros.fit_resample(np.array(X.values),np.array(y.values).reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X_r,y_r, test_size=0.30, random_state=42)
    
    y_test_fake = pd.DataFrame(X_test[:,1:2], columns=["is_fake_news"])
    X_test = X_test[:,:1]
    X_train = X_train[: , :1]
    
    
    train_data = pd.concat([pd.DataFrame(X_train, columns=['text']),
    pd.DataFrame(y_train, columns=['is_sarcasm'])], axis=1).reset_index()

    test_data = pd.concat([pd.DataFrame(X_test, columns=['text']),
    pd.DataFrame(y_test, columns=['is_sarcasm'])], axis=1).reset_index()

    train_data.to_csv("train.csv")
    test_data.to_csv("test.csv")
    y_test_fake.to_csv("fake_test.csv")

    return train_data.copy(), test_data.copy()
files = os.listdir('.')

if "train.csv" not in files and "test.csv" not in files:
    random_over_sample(clean_frame(read()))

train, test = pd.read_csv("train.csv"), pd.read_csv("test.csv")

y_pred_svm = congru.get_svm_explicit(train, test)
y_pred_mnb = congru.get_mnb_explicit(train, test)

svm_pred = pd.DataFrame(y_pred_svm, columns=['is_sarcasm'])
mnb_pred = pd.DataFrame(y_pred_mnb, columns=['is_sarcasm'])

y_fake = pd.read_csv("fake_test.csv")
utils.print_contingency_table(pd.concat([y_fake,svm_pred], axis = 1), "svm")
utils.print_contingency_table(pd.concat([y_fake,mnb_pred], axis = 1), "mnb")

utils.print_phi_inflated(pd.concat([y_fake,svm_pred], axis = 1), "svm")
utils.print_phi_inflated(pd.concat([y_fake,mnb_pred], axis = 1), "mnb")
