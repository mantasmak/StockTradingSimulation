#Mantas Maksimavicius PS 5 gr. 3 uzd.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

def load_to_dataframe_daily(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')
    df = df.set_index(df['Date'])
    df = df.drop(columns=['Date'])
    return df

def plot_graph(optimal_profit, nonoptimal_profit):
    plt.figure(figsize=(15, 6))
    subplt = plt.subplot(2, 1, 1)
    plt.plot(optimal_profit.cumsum())
    plt.title('Optimized')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    subplt = plt.subplot(2, 1, 2)
    plt.plot(nonoptimal_profit.cumsum())
    plt.title('Non-Optimized')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def aroon_up(data, period):
    period = int(period)

    a_up = [((period -
              list(reversed(data[idx + 1 - period:idx + 1])).index(np.max(data[idx + 1 - period:idx + 1]))) /
             float(period)) * 100 for idx in range(period - 1, len(data))]
    a_up = fill_for_noncomputable_vals(data, a_up)
    return a_up

def aroon_down(data, period):
    period = int(period)

    a_down = [((period -
                list(reversed(data[idx + 1 - period:idx + 1])).index(np.min(data[idx + 1 - period:idx + 1]))) /
               float(period)) * 100 for idx in range(period - 1, len(data))]
    a_down = fill_for_noncomputable_vals(data, a_down)
    return a_down

def fill_for_noncomputable_vals(input_data, result_data):
    non_computable_values = np.repeat(
        np.nan, len(input_data) - len(result_data)
        )
    filled_result_data = np.append(non_computable_values, result_data)
    return filled_result_data

def aroon_strategy(df, period, fee):
    a_up = aroon_up(df['High'], period)
    a_down = aroon_down(df['Low'], period)

    profit = pd.Series(np.zeros(df['Close'].size), index=df.index)
    position = 0
    for i in range(period, len(df['Close'])):
        profit[i] = (df['Close'][i] - df['Close'][i-1]) * position
        if a_up[i] >= a_down[i] and a_up[i-1] < a_down[i-1]:
            position = 1
            profit[i] -= fee * df['Close'][i]
        if a_up[i] < a_down[i] and a_up[i-1] >= a_down[i-1]:
            position = -1
            profit[i] -= fee * df['Close'][i]
    return profit

def sharpe(returns, N):
    return np.sqrt(N) * returns.mean() / returns.std()

register_matplotlib_converters()
df = load_to_dataframe_daily("AAPL_10_year.csv")

period = 1
fee = .005
optimal = pd.DataFrame(columns=['period', 'sharpe', 'returns'])
for period in range(2, 100, 20):
    print('period = {}'.format(period))
    returns = aroon_strategy(df, period, fee)
    sharpe_ratio = sharpe(returns, 252)
    optimal = optimal.append({'period': period, 'sharpe': sharpe_ratio, 'returns': returns.copy()}, ignore_index=True)

max_sharpe = optimal['sharpe'].idxmax()
print('Sharpe ratio ({}), Period {}'.format(optimal.loc[max_sharpe]['sharpe'], optimal.loc[max_sharpe]['period']))

plot_graph(optimal.loc[max_sharpe]['returns'], optimal.loc[0]['returns'])