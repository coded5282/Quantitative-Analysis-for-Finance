import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import os
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests

style.use('ggplot')

# use beautiful to get the source of the wiki page,
# find the table and load all the stock tickers from that page
# then save it in a pickle file
# so can read from that in the future
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers, f)
        
    print(tickers)
        
    return tickers

#save_sp500_tickers()

# load all sp500 stock data into csvs from google
def get_data_from_google(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)
            
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
        
    start = dt.datetime(2000,1,1)
    end = dt.datetime(2016,12,31)
    
    for ticker in tickers:
        try:
            print(ticker)
            if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                df = web.DataReader(ticker, 'google', start, end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            else:
                print('Already have {}'.format(ticker))
        except:
            print('Cannot obtain data for' + ticker)
            
#get_data_from_google()

# combine all stock data files into one file with just the close data
def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)
        
    main_df = pd.DataFrame()
    
    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
            
            df.rename(columns={'Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Volume'], 1, inplace=True)
            
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        except:
            print('stock_dfs/{}.csv'.format(ticker) + 'not found')
            
        if count % 10 == 0:
            print(count)
            
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')
    
#compile_data()

# plot a correlation heatmap b/w all companies in sp500
def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    #df['AAPL'].plot()
    #plt.show()
    # get correlations b/w stock close prices
    df_corr = df.corr()
    print(df_corr.head())
    # get just the values of the corr table
    data = df_corr.values
    fig = plt.figure()
    # add one plot
    ax = fig.add_subplot(1,1,1)
    # heatmap is red, yellow green
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    # building the heatmap by arranging ticks at every 1/2 mark
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    # removes empty space on top of plot
    ax.invert_yaxis()
    # moves x axis ticks to the top
    ax.xaxis.tick_top()
    
    # getting all ticker symbols
    column_labels = df_corr.columns
    row_labels = df_corr.index
    # setting the labels to the plot
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    # setting limit of colors
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()
    
visualize_data()