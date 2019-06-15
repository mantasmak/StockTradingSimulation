#Mantas Maksimavicius PS 5 gr. 2 uzd.

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

def plot_indicators(df):
    plt.figure(figsize=(15, 6))
    plt.subplot(4, 1, 1)
    plt.plot(df.index[25:], df['Close'][25:])
    plt.title('Alphabet daily stock')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.subplot(4, 1, 2)
    plt.plot(df.index[25:], aroon_up(df['High'], 25)[25:], color='g')
    plt.plot(df.index[25:], aroon_down(df['Low'], 25)[25:], color='r')
    plt.title('Aroon')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.subplots_adjust(top = 2)
    plt.grid(True)
    plt.subplot(4, 1, 3)
    plt.plot(df.index[25:], money_flow_index(df['Close'], df['High'], df['Low'], df['Volume'], 14)[25:], color='black')
    plt.title('Money Flow Index')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.subplots_adjust(top=2)
    plt.grid(True)
    plt.subplot(4, 1, 4)
    plt.plot(df.index[25:], on_balance_volume(df['Close'], df['Volume'])[25:], color='black')
    plt.title('On-Balance Volume')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.subplots_adjust(top=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def aroon_up(data, period):
    up = np.zeros(len(data) - period)
    for i in range(0, len(data) - period):
        period_data = list(reversed(data[i:period + i]))
        max_index = period_data.index(np.max(period_data))
        up[i] = ((period - max_index) / period) * 100

    nans = np.repeat(0, len(data) - len(up))
    up = np.append(nans, up)
    return up

def aroon_down(data, period):
    down = np.zeros(len(data) - period)
    for i in range(0, len(data) - period):
        period_data = list(reversed(data[i:period + i]))
        max_index = period_data.index(np.min(period_data))
        down[i] = ((period - max_index) / period) * 100

    nans = np.repeat(0, len(data) - len(down))
    down = np.append(nans, down)
    return down

def money_flow_index(close, high, low, volume, period):
    tp = np.zeros(len(close))
    for i in range(0, len(close)):
        tp[i] = (high[i] + low[i] + close[i]) / 3

    mf = volume * tp

    flow = np.zeros(len(tp))
    for i in range(1, len(tp)):
        flow[i - 1] = tp[i] > tp[i - 1]

    pf = np.zeros(len(flow))
    nf = np.zeros(len(flow))
    for i in range(0, len(flow)):
        if flow[i]:
            pf[i] = mf[i]
        else:
            pf[i] = 0

        if not flow[i]:
            nf[i] = mf[i]
        else:
            nf[i] = 0

    pmf = np.zeros(len(flow))
    nmf = np.zeros(len(flow))
    for i in range(period - 1, len(flow)):
        pmf[i] = sum(pf[i + 1 - period:i + 1])
        nmf[i] = sum(nf[i + 1 - period:i + 1])

    money_ratio = np.array(pmf) / np.array(nmf)

    mfi = 100 - (100 / (1 + money_ratio))

    nans = np.repeat(0, len(close) - len(mfi))
    mfi = np.append(nans, mfi)

    return mfi

def on_balance_volume(close, volume):
    obv = np.zeros(len(volume))
    for i in range(1, len(obv)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        elif close[i] == close[i-1]:
            obv[i] = obv[i-1]
    return obv

register_matplotlib_converters()
df = load_to_dataframe_daily('GOOG.csv')
plot_indicators(df)
print(df)