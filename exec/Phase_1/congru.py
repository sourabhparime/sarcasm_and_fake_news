# Functions to add explicit and implict congruity to the set of existing features
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import FunctionTransformer
import congruency_features as congo
import utils



 

def get_svm_explicit(train, test):

    
    #tweets = pd.concat([train, test], ignore_index = True, levels = None, keys = None)
    x_train, y_train, x_test, y_test = train.text.values.astype('U'), train.is_sarcasm.values, test.text.values.astype('U'), test.is_sarcasm.values
    #print(x_train)

    classifier = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
        ])),
        ('length', Pipeline([
            ('count', FunctionTransformer(congo.get_text_length, validate=False)),
        ])),
        ('pos_words', Pipeline([
            ('count', FunctionTransformer(congo.get_pos_words, validate=False)),
        ])),
        ('neg_words', Pipeline([
            ('count', FunctionTransformer(congo.get_neg_words, validate=False)),
        ])),
        ('comp', Pipeline([
            ('count', FunctionTransformer(congo.get_compound, validate=False)),
        ])),
        ('flips', Pipeline([
            ('count', FunctionTransformer(congo.get_number_of_flips, validate=False)),
        ])),
        ('sentiment_flips', Pipeline([
            ('count', FunctionTransformer(congo.get_sentiment_flip, validate=False)),
        ]))
    ])),
    ('clf', OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='rbf')))])

    classifier.fit(x_train, y_train)
    y_pred =  classifier.predict(x_test)
    utils.print_model_name("SVM with Congruency Features")
    utils.print_statistics(y_test, y_pred)
    utils.mislabelled_data_points(x_test, y_test, y_pred)

def get_mnb_explicit(train, test):

    x_train, y_train, x_test, y_test = train.text.values.astype('U'), train.is_sarcasm.values, test.text.values.astype('U'), test.is_sarcasm.values
    

    classifier = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
        ])),
        ('length', Pipeline([
            ('count', FunctionTransformer(congo.get_text_length, validate=False)),
        ])),
        ('pos_words', Pipeline([
            ('count', FunctionTransformer(congo.get_pos_words, validate=False)),
        ])),
        ('neg_words', Pipeline([
            ('count', FunctionTransformer(congo.get_neg_words, validate=False)),
        ])),
        ('comp', Pipeline([
            ('count', FunctionTransformer(congo.get_compound_mnb, validate=False)),
        ])),
        ('flips', Pipeline([
            ('count', FunctionTransformer(congo.get_number_of_flips, validate=False)),
        ])),
        ('sentiment_flips', Pipeline([
            ('count', FunctionTransformer(congo.get_sentiment_flip, validate=False)),
        ]))
    ])),
    ('clf', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))])

    classifier.fit(x_train, y_train)
    y_pred =  classifier.predict(x_test)
    utils.print_model_name("Multinomial Naive Bayes with Congruency Features")
    utils.print_statistics(y_test, y_pred)
    utils.mislabelled_data_points(x_test, y_test, y_pred)