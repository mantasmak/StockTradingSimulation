#Mantas Maksimavicius PS 5 gr. 3 uzd.

import sys
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

def plot_graph(df, a_up, a_down, crossovers, profit, take_profit, stop_loss):
    plt.figure(figsize=(15, 6))
    subplt1 = plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'])
    plt.plot(df.index[crossovers], df['Close'][crossovers], 'ro')
    plt.plot(df.index[take_profit], df['Close'][take_profit], 'go')
    plt.plot(df.index[stop_loss], df['Close'][stop_loss], 'bo')
    plt.title('Apple daily stock')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.subplot(3, 1, 2, sharex=subplt1)
    plt.plot(df.index, a_up, color='g')
    plt.plot(df.index, a_down, color='r')
    plt.title('Aroon')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.subplot(3, 1, 3, sharex=subplt1)
    plt.plot(df.index, profit.cumsum())
    plt.title('Profit')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.subplots_adjust(top=2)
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

def find_crossovers(a_up, a_down, date_index):
    arr_up = pd.Series(a_up, index=date_index)
    arr_down = pd.Series(a_down, index=date_index)
    up_greater = arr_up > arr_down
    crossovers = up_greater != up_greater.shift(1)
    return crossovers

def aroon_strategy(df, period, fee, take_profit_limit, stop_loss_limit):
    a_up = aroon_up(df['High'], period)
    a_down = aroon_down(df['Low'], period)
    entry_profit = 0
    take_profit = pd.Series(False, index=df.index)
    stop_loss = pd.Series(False, index=df.index)

    profit = pd.Series(np.zeros(df['Close'].size), index=df.index)
    position = 0
    for i in range(period, len(df['Close'])):
        profit[i] = (df['Close'][i] - df['Close'][i-1]) * position
        if a_up[i] >= a_down[i] and a_up[i-1] < a_down[i-1]:
            position = 1
            profit[i] -= fee * df['Close'][i]
            entry_profit = profit[i]
        if a_up[i] < a_down[i] and a_up[i-1] >= a_down[i-1]:
            position = -1
            profit[i] -= fee * df['Close'][i]
            entry_profit = profit[i]
        if position != 0 and profit[i] - entry_profit > take_profit_limit:
            position = 0
            take_profit[i] = True
        if position != 0 and entry_profit - profit[i] > stop_loss_limit:
            position = 0
            stop_loss[i] = True
    return profit, take_profit, stop_loss

register_matplotlib_converters()
df = load_to_dataframe_daily("AAPL_10_yearx .csv")
a_up = aroon_up(df['High'], 60)
a_down = aroon_down(df['Low'], 60)
crossovers = find_crossovers(a_up, a_down, df.index)
profit, take_profit, stop_loss = aroon_strategy(df, 60, .005, 10, 6)
plot_graph(df, a_up, a_down, crossovers, profit, take_profit, stop_loss)
#print(crossovers)