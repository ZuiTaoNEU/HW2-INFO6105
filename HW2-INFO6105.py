import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download stock data
stock_symbol = 'AAPL'  # You can replace 'AAPL' with any stock symbol
start_date = '2023-01-01'
end_date = '2023-12-31'
stock = yf.download(stock_symbol, start=start_date, end=end_date)

# 1. Calculate Simple Moving Average (SMA)
stock['SMA_50'] = stock['Adj Close'].rolling(window=50).mean()  # 50-day SMA
stock['SMA_200'] = stock['Adj Close'].rolling(window=200).mean()  # 200-day SMA

# 2. Calculate Relative Strength Index (RSI)
def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

stock['RSI'] = calculate_rsi(stock['Adj Close'])

# 3. Calculate On-Balance Volume (OBV)
stock['OBV'] = (np.sign(stock['Adj Close'].diff()) * stock['Volume']).cumsum()

# Plotting the indicators
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Plot Adjusted Close Price with SMA
ax1.plot(stock.index, stock['Adj Close'], label='Adjusted Close Price', color='blue')
ax1.plot(stock.index, stock['SMA_50'], label='50-Day SMA', color='orange', linestyle='--')
ax1.plot(stock.index, stock['SMA_200'], label='200-Day SMA', color='green', linestyle='--')
ax1.set_title(f'{stock_symbol} Stock Price with SMA')
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
