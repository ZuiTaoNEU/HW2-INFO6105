import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download stock data
stock_symbol = 'PANW'
start_date = '2023-01-01'
end_date = '2024-09-30'
stock = yf.download(stock_symbol, start=start_date, end=end_date)

# 1. Calculate Simple Moving Average (SMA)
stock['SMA_50'] = stock['Adj Close'].rolling(window=50).mean()  # 50-day SMA
stock['SMA_200'] = stock['Adj Close'].rolling(window=200).mean()  # 200-day SMA

# 2. Calculate Relative Strength Index (RSI) with a 21-day period
def calculate_rsi(series, period=21):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

stock['RSI'] = calculate_rsi(stock['Adj Close'], period=21)

# 3. Calculate On-Balance Volume (OBV)
stock['OBV'] = (np.sign(stock['Adj Close'].diff()) * stock['Volume']).cumsum()

# 4. Generate Buy Signals based on RSI + OBV and SMA + OBV
stock['RSI_OBV_Buy'] = (stock['RSI'] < 30) & (stock['RSI'].shift(1) < stock['RSI']) & (stock['OBV'] > stock['OBV'].shift(1))
stock['SMA_OBV_Buy'] = ((stock['Adj Close'] > stock['SMA_50']) | (stock['Adj Close'] > stock['SMA_200'])) & (stock['OBV'] > stock['OBV'].shift(1))

# Plotting the indicators and buy signals
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

# Plot Adjusted Close Price with SMA and buy signals
ax1.plot(stock.index, stock['Adj Close'], label='Adjusted Close Price', color='blue')
ax1.plot(stock.index, stock['SMA_50'], label='50-Day SMA', color='orange', linestyle='--')
ax1.plot(stock.index, stock['SMA_200'], label='200-Day SMA', color='green', linestyle='--')
# Highlight RSI + OBV buy signals
ax1.plot(stock[stock['RSI_OBV_Buy']].index, stock['Adj Close'][stock['RSI_OBV_Buy']], '^', markersize=10, color='purple', label='RSI + OBV Buy Signal')
# Highlight SMA + OBV buy signals
ax1.plot(stock[stock['SMA_OBV_Buy']].index, stock['Adj Close'][stock['SMA_OBV_Buy']], '^', markersize=10, color='red', label='SMA + OBV Buy Signal')
ax1.set_title(f'{stock_symbol} Stock Price with SMA and Buy Signals')
ax1.set_ylabel('Price')
ax1.legend(loc='upper left')

# Plot RSI
ax2.plot(stock.index, stock['RSI'], label='RSI', color='purple')
ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
ax2.set_title(f'{stock_symbol} RSI (Relative Strength Index)')
ax2.set_ylabel('RSI')
ax2.legend(loc='upper left')

# Plot OBV
ax3.plot(stock.index, stock['OBV'], label='OBV', color='brown')
ax3.set_title(f'{stock_symbol} On-Balance Volume (OBV)')
ax3.set_ylabel('OBV')
ax3.legend(loc='upper left')

plt.xlabel('Date')
plt.show()
