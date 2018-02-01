import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

# preprocess data to get % price change
def process_data_for_labels(ticker):
    # number of days we observe for decision
    hm_days = 7
    # load data
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    # get all tickers in sp500 into a list
    tickers = df.columns.values.tolist()
    # fill any nans with 0
    df.fillna(0, inplace=True)
    
    # loop for # of days
    for i in range(1, hm_days+1):
        # getting the % change in i days by shifting the dataframe
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    
    df.fillna(0, inplace=True)
    return tickers, df

#process_data_for_labels('XOM')

# return buy, sell, or hold signal
def buy_sell_hold(*args):
    cols = [c for c in args]
    # stock price change percentage
    requirement = 0.02
    for col in cols:
        if col > requirement:
            # buy
            return 1
        if col < -requirement:
            # sell
            return -1
    # hold
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                      df['{}_1d'.format(ticker)],
                                      df['{}_2d'.format(ticker)],
                                      df['{}_3d'.format(ticker)],
                                      df['{}_4d'.format(ticker)],
                                      df['{}_5d'.format(ticker)],
                                      df['{}_6d'.format(ticker)],
                                      df['{}_7d'.format(ticker)],
                                      ))
    
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))
    df.fillna(0, inplace=True)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
          
    return X, y, df

def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=.25)
    
    #clf = neighbors.KNeighborsClassifier()
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy', confidence)    
    predictions = clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))
    
    return confidence

do_ml('BAC')