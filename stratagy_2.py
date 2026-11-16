import MetaTrader5 as mt5
import time
import pandas as pd
import numpy as np
import gc
import math
from datetime import datetime,time as tm
from order import close_all,buy,sell,loss_manage,info,close_by_symbol
# import threading
#This Stratagy base on: Supertrend+ATR+Session+EMA Cross

try:    
    try:
        mt5.initialize()
        print("Metatrader is open") 
           
    except:
        print(f"Metatrader opening is fail: {mt5.last_error()}")

    trade_limit=-400
    symbol="BTCUSDm"
    bar_count=500
    fast_len=50
    slow_len=200
    lot=0.01
    plan=1
    timefram=mt5.TIMEFRAME_M5
    multiplier=2.5
    length=10

    def calculate_ema_numpy(prices: np.ndarray, length: int) -> np.ndarray:
        """
        Calculates pure exponential moving average using NumPy.
        Matches TradingView / Exness engine logic.
        """
        n = len(prices)
        ema = np.empty(n, dtype=np.float64)

        # Initialize seed at first available bar
        ema[0] = prices[0]
        alpha = 2.0 / (length + 1.0)

        # Vectorized exponential calculation
        for i in range(1, n):
            ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1.0 - alpha))
            
        return ema
 
    def calculate_required_bars(ema_length: int, precision: float = 0.999) -> int:
        """
        Calculates required historical bars (Y) for a given EMA length (X).

        :param ema_length: EMA period (X)
        :param precision: Target accuracy (default 0.999 for 99.9% accuracy)
        :return: Minimum number of bars (Y) required
        """
        alpha = 2.0 / (ema_length + 1.0)
        required_y = math.log(1.0 - precision) / math.log(1.0 - alpha)

        total_bars = math.ceil(required_y) + ema_length
        return total_bars

    def calculate_exness_ema_cross(df: pd.DataFrame,fast_len: int = 20,slow_len: int = 300) -> pd.DataFrame:
        """
        Calculates Fast & Slow EMAs and detects cross signals.

        Signal Outputs:
            1  = BULLISH CROSSOVER (Fast crossed ABOVE Slow)
            -1  = BEARISH CROSSOVER (Fast crossed BELOW Slow)
            0  = NO CROSSOVER (Trend continuation)
        """
        close_prices = df['close'].values.astype(np.float64)

        # 1. Calculate Fast and Slow EMAs
        fast_ema = calculate_ema_numpy(close_prices, fast_len)
        slow_ema = calculate_ema_numpy(close_prices, slow_len)

        # 2. Determine ongoing trend state (+1 for Fast > Slow, -1 for Fast < Slow)
        trend = np.where(fast_ema > slow_ema, 1, -1)

        # 3. Detect crossover events (Compare current trend vs previous bar's trend)
        prev_trend = np.roll(trend, 1)
        prev_trend[0] = trend[0]  # First element baseline

        # Crossover occurs when trend changes relative to prior bar
        crossover_signal = np.where(trend != prev_trend, trend, 0)

        df['fast_ema'] = fast_ema
        df['slow_ema'] = slow_ema
        df['trend'] = trend
        df['cross_signal'] = crossover_signal
        last_bar = df.iloc[-1]
        
        return last_bar

    def calculate_supertrend_raw(df, length, multiplier):
        """
        Calculates Supertrend using pure Pandas and NumPy.
        Calculates Supertrend matching standard platform (Exness/TradingView) engines.
        """
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        n = len(df)
        
        # 2. True Range (TR) Calculation
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # 3. Wilder's Exponential Smoothing (Alpha = 1 / length)
        atr = pd.Series(tr).ewm(alpha=1/length, adjust=False).mean().values

        # 4. Median Price Baseline (HL2)
        hl2 = (high + low) / 2.0
        basic_ub = hl2 + (multiplier * atr)
        basic_lb = hl2 - (multiplier * atr)

        # 5. Recursive Band Trailing State Loop
        final_ub = np.zeros(n)
        final_lb = np.zeros(n)
        supertrend = np.zeros(n)
        direction = np.zeros(n, dtype=int)

        for i in range(1, n):
            # Upper Trailing Stop
            if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]

            # Lower Trailing Stop
            if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]

            # Direction Switching
            if direction[i-1] == 1:
                if close[i] < final_lb[i]:
                    direction[i] = -1
                    supertrend[i] = final_ub[i]
                else:
                    direction[i] = 1
                    supertrend[i] = final_lb[i]
            else:
                if close[i] > final_ub[i]:
                    direction[i] = 1
                    supertrend[i] = final_lb[i]
                else:
                    direction[i] = -1
                    supertrend[i] = final_ub[i]

        df['supertrend'] = supertrend
        df['signal'] = direction
        
        """
        Extracts signal output from the last completed candle (iloc[-1]).
        Returns signal = 1 for Green line and signal = -1 for Red line.
        """
        last_bar = df.iloc[-1]
        
        return {
            "timestamp": df.index[-1],
            "close": last_bar['close'],
            "supertrend_line": last_bar['supertrend'],
            "signal": int(last_bar['signal']),  # 1 (Green / Bullish) or -1 (Red / Bearish)
            "action": "BULLISH" if last_bar['signal'] == 1 else "BEARISH"
            
        }
                
    def calculate_chop(df: pd.DataFrame, length: int ,upper_threshold:float) -> pd.Series:
        """
        Calculates Choppiness Index (CHOP).
        Returns:
        is_sideways = 1 -> Market is CHOPPY / SIDEWAYS (Chop >= 61.8)
        is_sideways = 0 -> Market is TRENDING        (Chop < 61.8)
        """
        df = df.copy()

        # 1. True Range (TR)
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 2. Rolling ATR sum over N bars
        atr_sum = tr.rolling(window=length).sum()

        # 3. Highest High minus Lowest Low over N bars
        high_max = high.rolling(window=length).max()
        low_min = low.rolling(window=length).min()
        range_max_min = high_max - low_min

        # 4. Standard Choppiness Index Formula (0 to 100)
        # Avoid division by zero with np.where
        ratio = np.where(range_max_min > 0, atr_sum / range_max_min, 1.0)
        chop = 100.0 * (np.log10(ratio) / np.log10(length))
        
        df['chop'] = chop

        # 5. Clean NaN values from warm-up period
        df['chop'] = df['chop'].bfill().fillna(50.0)

        # 6. Strict Binary Flag (1 = Sideways/Choppy, 0 = Trending)
        df['is_sideways'] = np.where(df['chop'] >= upper_threshold, 1, 0)
        return df       

    while True:
        start=time.perf_counter()
        bar_count_e=calculate_required_bars(ema_length=slow_len,)
        rates= mt5.copy_rates_from_pos(symbol, timefram, 0, bar_count)
        rates_e= mt5.copy_rates_from_pos(symbol, timefram, 1, bar_count_e)
        df = pd.DataFrame(rates_e)
        last_bar=calculate_exness_ema_cross(df,fast_len=fast_len,slow_len=slow_len)
        df=pd.DataFrame(rates)
        back=calculate_supertrend_raw(df,length,multiplier)
        info_re=info(symbol)
        calculate_chops = calculate_chop(df, length=14, upper_threshold=56)
        print(calculate_chops[['close', 'chop', 'is_sideways']].iloc[-1])
        # action=loss_manage(trade_limit) #Calculate the loss management
        if calculate_chops['is_sideways'].iloc[-1]==0 :
            print('pass 1')
        # if calculate_chops['is_sideways'].iloc[-1]==0 and action:
            if back["signal"] == 1 and (info_re==0 or info_re==-1 ) and last_bar['fast_ema']>=last_bar['slow_ema']:
                if datetime.now().time()>=tm(18,30,0) or datetime.now().time()<=tm(4,30,0):
                # if True:
                    print(f"signal:  {back['signal']}  form buy")
                    print(f"infor for buy {info_re}")
                    close_all()
                #    close_by_symbol(symbol=symbol,entry=-1)  #Facing issue on closing 
                    ticket=buy(symbol, lot,back["supertrend_line"])
                    print(f"BUY at {back["close"]}")
                else:
                    close_all()
                # last_sl=back["supertrend_line"]
            elif back["signal"] == -1 and (info_re==0 or info_re==1 ) and last_bar['fast_ema']<=last_bar['slow_ema']:
                print('pass 2')
                if datetime.now().time()>=tm(18,30,0) or datetime.now().time()<=tm(4,30,0) :
                # if True:
            # else:                    
                    print(f"signal:  {back['signal']} for sell")
                    print(f"infor for sell {info_re}")
                    print('pass 2')
                #   close_by_symbol(symbol=symbol,entry=1) #Facing issue on closing 
                    close_all()
                    ticket=sell(symbol,lot,back["supertrend_line"])
                    print(f"SELL at {back["supertrend_line"]}")
                # last_sl=back["supertrend_line"]
                else:
                    close_all()
        else:
            print(f"No Action taken")
        # if back["supertrend_line"]!=last_sl and last_sl != 0:
        #     modify(tickets=ticket,change=back["supertrend_line"])
        
        
        print(f"Latest Close Price:  {last_bar['close']:.2f}")
        print(f"Fast EMA ({fast_len}):       {last_bar['fast_ema']:.2f}")
        print(f"Slow EMA ({slow_len}):      {last_bar['slow_ema']:.2f}")
        print(f"Current Trend State: {'BULLISH (+1)' if last_bar['trend'] == 1 else 'BEARISH (-1)'}")
        print(f"Crossover Event:     {int(last_bar['cross_signal'])}")
        print(f"close: {back["close"]}")
        print(f"supertrend_line: {back["supertrend_line"]}")
        print(f"action: {back["action"]}")
        print(f"market direction:{back["signal"]}")
        print(f"is sideways:{calculate_chops['is_sideways'].iloc[-1]!=0}")
        end = time.perf_counter()
        total=end-start
        print(f"Time: {total}")
        del df
        del back
        del rates
        gc.collect()
        # time.sleep(60)
        # time.sleep(5)
        time.sleep(5)
        # break

except KeyboardInterrupt:
    close_by_symbol(symbol=symbol,entry=0)#type:ignore
    print("All Position is Close")
