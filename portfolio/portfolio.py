#Mantas Maksimavicius PS 5 gr. 3 uzd.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import random
import glob
import seaborn as sns

def read_files():
    stock_dict = {}
    path = '*_10_year.csv'
    files = glob.glob(path)
    for name in files:
        stock_name = name.rstrip('_10_year.csv')
        stock_dict[stock_name] = load_to_dataframe_daily(name)
    return stock_dict

def load_to_dataframe_daily(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')
    df = df.set_index(df['Date'])
    df = df.drop(columns=['Date'])
    return df

def plot_graph(optimal_profit, corr_table):
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.plot(optimal_profit.cumsum())
    plt.title('Portfolio profit')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.title('Correlation matrix')
    sns.heatmap(corr_table, annot=True)
    plt.tight_layout()
    plt.show()
    plt.close()

def aroon_up(values, period):
    period = int(period)

    a_up = [((period -
              list(reversed(values[idx + 1 - period:idx + 1])).index(np.max(values[idx + 1 - period:idx + 1]))) /
             float(period)) * 100 for idx in range(period - 1, len(values))]
    a_up = fill_for_noncomputable_vals(values, a_up)
    return a_up

def aroon_down(values, period):
    period = int(period)

    a_down = [((period -
                list(reversed(values[idx + 1 - period:idx + 1])).index(np.min(values[idx + 1 - period:idx + 1]))) /
               float(period)) * 100 for idx in range(period - 1, len(values))]
    a_down = fill_for_noncomputable_vals(values, a_down)
    return a_down

def fill_for_noncomputable_vals(input_data, result_data):
    non_computable_values = np.repeat(
        np.nan, len(input_data) - len(result_data)
        )
    filled_result_data = np.append(non_computable_values, result_data)
    return filled_result_data

def aroon_strategy(df, period, price):
    a_up = aroon_up(df['High'], period)
    a_down = aroon_down(df['Low'], period)

    earnings = pd.Series(np.zeros(df['Close'].size), index=df.index)
    position = 0
    for it in range(period, len(df['Close'])):
        earnings[it] = (df['Close'][it] - df['Close'][it-1]) * position
        if a_up[it] >= a_down[it] and a_up[it-1] < a_down[it-1]:
            position = 1
            earnings[it] -= price * df['Close'][it]
        if a_up[it] < a_down[it] and a_up[it-1] >= a_down[it-1]:
            position = -1
            earnings[it] -= price * df['Close'][it]
    return earnings

def sharpe(ret, n):
    return np.sqrt(n) * ret.mean() / ret.std()

def convert_series_to_dataframe(stock_name, optimized):
    max_sharpe = optimized[stock_name]['sharpe'].idxmax()
    df_to_add = pd.DataFrame({'Date': optimized[stock_name].loc[max_sharpe]['returns'].index,
                              'Profit': optimized[stock_name].loc[max_sharpe]['returns'].values})
    df_to_add = df_to_add.set_index(df_to_add['Date'])
    df_to_add = df_to_add.drop(columns=['Date'])
    return df_to_add

register_matplotlib_converters()
data = read_files()

# Optimalizacija
fee = .005
optimal = {}
for stock in data:
    optimal[stock] = pd.DataFrame(columns=['period', 'sharpe', 'returns'])
    for i in range(0, 10):#range(1, 100, 2):
        per = random.randint(5, 80)
        print('{}: period = {}'.format(stock, per))
        returns = aroon_strategy(data[stock], per, fee)
        sharpe_ratio = sharpe(returns, 252)
        optimal[stock] = optimal[stock].append({'period': per, 'sharpe': sharpe_ratio, 'returns': returns.copy()}, ignore_index=True)

# Sujungtas pelnas
profit = pd.DataFrame(index=data['AMZN'].index, columns=['Profit'])
profit['Profit'] = 0.0
for stock in data:
    profit_to_add = convert_series_to_dataframe(stock, optimal)
    profit = profit + profit_to_add

# Koreliacija
corr_dict = {'Stock1':[], 'Stock2':[], 'Correlation':[]}
for stock1 in data:
    for i in range(0, 10):
        corr_dict['Stock1'].append(stock1)
    max_sharpe1 = optimal[stock1]['sharpe'].idxmax()
    for stock2 in data:
        corr_dict['Stock2'].append(stock2)
        max_sharpe2 = optimal[stock2]['sharpe'].idxmax()
        if stock1 != stock2:
            corr_dict['Correlation'].append(optimal[stock1].loc[max_sharpe1]['returns'].corr(optimal[stock2].loc[max_sharpe2]['returns']))
        else:
            corr_dict['Correlation'].append(float('nan'))

# Grafikai
corr_df = pd.DataFrame(corr_dict)
table = corr_df.pivot(index='Stock1', columns='Stock2', values='Correlation')
plot_graph(profit, table)