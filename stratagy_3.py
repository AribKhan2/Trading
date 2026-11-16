import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import time
from order import close_all,buy,sell,loss_manage,info,close_by_symbol
import gc
#This stratagy base on RSI 70/30 system
try:
    try:
        mt5.initialize()
        print("Metatrader is open")    
    except:
        print(f"Metatrader opening is fail: {mt5.last_error()}")

    symbol="BTCUSDm"
    bar_count=500
    rsi_length=14
    timeframe=mt5.TIMEFRAME_M5
    lot=0.01
    trade_limit=-400
    # ticket=0


    def calculate_rsi_numpy(prices: np.ndarray, length: int = 14) -> np.ndarray:
        """
        Calculates Relative Strength Index (RSI) using Wilder's Smoothing via NumPy.
        Matches TradingView / MT5 standard calculations exactly.
        
        :param prices: 1D NumPy array of close prices
        :param length: Lookback period (default: 14)
        :return: 1D NumPy array with calculated RSI values
        """
        n = len(prices)
        rsi = np.full(n, np.nan, dtype=np.float64)
        
        if n <= length:
            return rsi

        # 1. Calculate price changes (deltas)
        deltas = np.diff(prices)
        
        # 2. Separate gains (positive changes) and losses (negative changes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Arrays to store smoothed averages
        avg_gains = np.zeros(n - 1, dtype=np.float64)
        avg_losses = np.zeros(n - 1, dtype=np.float64)

        # 3. First value is the SMA of the first 'length' changes
        avg_gains[length - 1] = np.mean(gains[:length])
        avg_losses[length - 1] = np.mean(losses[:length])

        # 4. Apply Wilder's Smoothing across the rest of the array
        for i in range(length, len(deltas)):
            avg_gains[i] = (avg_gains[i - 1] * (length - 1) + gains[i]) / length
            avg_losses[i] = (avg_losses[i - 1] * (length - 1) + losses[i]) / length

        # 5. Compute Relative Strength (RS) and RSI
        # Avoid division by zero when losses are 0
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gains / avg_losses
            rsi_values = 100.0 - (100.0 / (1.0 + rs))
            
            # Handle cases where avg_loss is 0 (RSI = 100)
            rsi_values[avg_losses == 0] = 100.0

        # Align output length with original price array length (shifts by 1 due to np.diff)
        rsi[length:] = rsi_values[length - 1:]
        
        return rsi

    # if login_success
    while True:
        start=time.perf_counter()
        rates= mt5.copy_rates_from_pos(symbol, timeframe, 0, bar_count)
        df=pd.DataFrame(rates)
        print(df)
        # close_prices = rates['close'].astype(np.float64)
        # rsi = calculate_rsi_numpy(close_prices, length=rsi_length)
        # account_info = mt5.account_info()
        # print(f"Account Balance: {account_info.balance}")
        # print(f"Broker/Server: {account_info.server}")
        # print(f"Latest Bar Close: {close_prices[-1]:.2f}")
        # print(f"Latest bar RSI: {rsi[-1]:.2f}")
        # info_re = info(symbol=symbol)
        # # action=loss_manage(trade_limit)
        
        # action=True
        # if action:
        #     if rsi[-1] >= 70 and info_re == 0:
        #         # ticket=sell(symbol=symbol,lot=lot)
        #         print(f"Sell order {ticket}")    
        #     elif rsi[-1] <= 51 and info_re == -1:
        #         close_all()
        #         print('close all for sell')
        #     elif rsi[-1] <= 30 and info_re==0:
        #         # ticket=buy(symbol=symbol,lot=lot)
        #         print(f"Buy order {ticket}")
        #     elif rsi[-1] >= 49 and info_re==1:
        #         close_all()
        #         print('close all for buy')
        #     else:
        #         print('No Action Taken')
        # else:
        #         print("No Action Taken")
        
        # del rates
        # del close_prices
        # del rsi
        # gc.collect()
        # end= time.perf_counter()
        # total_time=end-start
        # print(f"time taken: {total_time}")
        # time.sleep(0.5)
        break

except KeyboardInterrupt:
    close_by_symbol(symbol=symbol,entry=0)#type:ignore
    print("All Position is Close")

        
            

