from sklearn import metrics
import numpy as np
import pandas as pd
import scipy.stats as scipy



def print_model_name(name):
    print("\n==================================================================")
    print('{:>20}'.format(name))
    print("==================================================================\n")

def print_statistics(y, y_pred):
    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred, average='weighted')
    recall = metrics.recall_score(y, y_pred, average='weighted')
    f_score = metrics.f1_score(y, y_pred, average='weighted')
    print('Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nF_score: %.3f\n'
          % (accuracy, precision, recall, f_score))
    print(metrics.classification_report(y, y_pred))
    return accuracy, precision, recall, f_score

def mislabelled_data_points(x_test, y_test, y_pred):
    count = 0
    for i in range(len(x_test)):
        num = y_pred[i]
        if num != y_test[i]:
            print('Expected:', y_test[i], ' but predicted ', num)
            print(x_test[i])
            count += 1
    print(count)

def print_contingency_table(df, model):
    print("====================================================")
    print("Printing contingency for "+ model )
    print("====================================================")
    print(pd.crosstab(df['is_fake_news'] == True, df['is_sarcasm'] == True))

def print_phi_inflated(df,model):
    print("==============================================")
    print("Printing pearson (inflated) for "+ model)
    print(scipy.pearsonr(df['is_sarcasm'].values,df['is_fake_news'].values)[0])
    print("==============================================")