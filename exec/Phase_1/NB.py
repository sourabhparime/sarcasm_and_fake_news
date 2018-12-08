from nltk import bigrams, trigrams
from nltk import NaiveBayesClassifier
from nltk import FreqDist
import nltk, random

def get_accuracy(sar, non_sar, test_tweets):
        
    def get_bigram_features(sar, non_sar):
        bigram_feature_vectors = []
        for tweet, tone in sar:
            temp = []
            
            for item in trigrams(tweet.split()):
                temp.append(item)
            bigram_feature_vectors.append((temp, tone))

        for tweet, tone in non_sar:
            temp = []
            #print(tweet)
            for item in bigrams(tweet.split()):
                temp.append(item)
            bigram_feature_vectors.append((temp, tone))
        return bigram_feature_vectors

    def get_unigram_features(sar, non_sar):
        bigram_feature_vectors = []
        for tweet, tone in sar:
            temp = []
            #print(tweet)
            temp.extend(tweet.split(" "))
            bigram_feature_vectors.append((temp, tone))

        for tweet, tone in non_sar:
            temp = []
            #print(tweet)
            temp.extend(tweet.split(" "))
            bigram_feature_vectors.append((temp, tone))
        return bigram_feature_vectors


    def get_word_features(bigram_feature_vectors):
        # get all words
        all_bigrams = []
        for (bigrams, tone) in bigram_feature_vectors:
            all_bigrams.extend(bigrams)
        
        wordlist = FreqDist(all_bigrams)
        word_features  = wordlist.keys()
        return word_features

    def extract_feature(doc):
        doc_words = set(doc)
        features = {}
        for word in word_features:
            #print(word)
            features['contains('+str(word)+')'] = (word in doc_words)
        return features

    bigram_feature_vectors = get_bigram_features(sar, non_sar)
    #bigram_feature_vectors = get_unigram_features(sar, non_sar)
    #print(bigram_feature_vectors)
    random.shuffle(bigram_feature_vectors)
    word_features = get_word_features(bigram_feature_vectors)
    #print(sar)
    training_set = nltk.classify.apply_features(extract_feature, bigram_feature_vectors)

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    #print(classifier.show_most_informative_features(32))
    preds = []
    for tweet in test_tweets:
        if classifier.classify(extract_feature(tweet[0].split())) == 'sarcastic':
            preds.append(True)
        else:
            preds.append(False)
    return preds












