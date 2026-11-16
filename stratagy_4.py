import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import math
import time
from order import close_all,buy,sell,loss_manage,info,close_by_symbol
#EMA Cross
try:
    try:
        mt5.initialize()
    except:
        print(f"Metatrader opening is fail: {mt5.last_error()}")

    symbol="BTCUSDm"
    timeframe = mt5.TIMEFRAME_M5
    lot=0.01
    fast_len=50
    slow_len=200
    # ticket=0
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


    while True:
        start=time.perf_counter()
        # mt5.login(
        # login=463946406,
        # password="240219#Arib",
        # server="Exness-MT5Trial17"
        # )
        bar_count= calculate_required_bars(ema_length=slow_len,precision=0.999)
        rates=mt5.copy_rates_from_pos(symbol,timeframe,1,bar_count)
        df=pd.DataFrame(rates)
        last_bar = calculate_exness_ema_cross(df, fast_len, slow_len)
        info_re=info(symbol)

        print(f"Latest Close Price:  {last_bar['close']:.2f}")
        print(f"Fast EMA ({fast_len}):       {last_bar['fast_ema']:.2f}")
        print(f"Slow EMA ({slow_len}):      {last_bar['slow_ema']:.2f}")
        print(f"Current Trend State: {'BULLISH (+1)' if last_bar['trend'] == 1 else 'BEARISH (-1)'}")
        print(f"Crossover Event:     {int(last_bar['cross_signal'])}")

        print(f'first: {last_bar['fast_ema']>=last_bar['slow_ema']}')
        print(f'second: {last_bar['fast_ema']<=last_bar['slow_ema']}')
        if int(last_bar['cross_signal']) ==1 and (info_re==0 or info_re==-1 ):
            close_all()
            ticket=buy(symbol=symbol,lot=lot)
            print(f"Buy order {ticket}") 
        elif int(last_bar['cross_signal']) ==-1 and (info_re==0 or info_re==1 ):
            close_all()
            ticket=sell(symbol=symbol,lot=lot)
            print(f"Sell order {ticket}")
        else:
            print("No Action Taken")




        del df
        del rates
        del last_bar
        del bar_count
        end=time.perf_counter()
        total_time=end-start
        print(f"time taken: {total_time}")
        time.sleep(1)
except KeyboardInterrupt:
    close_by_symbol(symbol=symbol,entry=0)#type:ignore
    print("All Position is Close")

    