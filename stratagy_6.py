import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from stratagy.order import close_by_symbol,buy,sell,info,notify_bot
#High frequency trading
#Market are btc and gold
symbol="BTCUSDm"
lot=0.01
try:
    try:
        mt5.initialize()
        print("Metatrader Opening")
    except:
        print(f"Metatrader Fail to Open: {mt5.last_error()}")

    def analyze_market_trend(df, left_bars=5, right_bars=5, num_pivots_to_check=3):
        """
        Detects market trend using pivot highs and pivot lows in a single function.
        
        Parameters:
        - df: pandas DataFrame containing 'High' and 'Low' columns.
        - left_bars: number of bars to the left to confirm a pivot.
        - right_bars: number of bars to the right to confirm a pivot.
        - num_pivots_to_check: number of recent pivots to analyze for trend structure.
        
        Returns:
        - trend: string describing the market trend.
        - ph_values: array of recent pivot high values.
        - pl_values: array of recent pivot low values.
        """
        highs = df['high'].values
        lows = df['low'].values
        
        pivot_highs = np.zeros(len(df), dtype=bool)
        pivot_lows = np.zeros(len(df), dtype=bool)
        
        # 1. Identify Pivot Points
        for i in range(left_bars, len(df) - right_bars):
            # Check Pivot High
            is_high = True
            for j in range(1, left_bars + 1):
                if highs[i] <= highs[i - j] or highs[i] < highs[i + j]:
                    is_high = False
                    break
            if is_high:
                pivot_highs[i] = True
                
            # Check Pivot Low
            is_low = True
            for j in range(1, left_bars + 1):
                if lows[i] >= lows[i - j] or lows[i] > lows[i + j]:
                    is_low = False
                    break
            if is_low:
                pivot_lows[i] = True
                
        df['Pivot_High'] = pivot_highs
        df['Pivot_Low'] = pivot_lows
        
        # 2. Extract recent pivot highs and lows
        ph = df[df['Pivot_High']].tail(num_pivots_to_check)
        pl = df[df['Pivot_Low']].tail(num_pivots_to_check)
        
        if len(ph) < 2 or len(pl) < 2:
            return "Insufficient Pivots for Trend Detection", np.array([]), np.array([])
            
        ph_values = ph['high'].values
        pl_values = pl['low'].values
        
        # 3. Check sequence structure (Higher Highs, Higher Lows, etc.)
        higher_highs = all(ph_values[i] > ph_values[i-1] for i in range(1, len(ph_values)))
        higher_lows = all(pl_values[i] > pl_values[i-1] for i in range(1, len(pl_values)))
        
        lower_highs = all(ph_values[i] < ph_values[i-1] for i in range(1, len(ph_values)))
        lower_lows = all(pl_values[i] < pl_values[i-1] for i in range(1, len(pl_values)))
        
        # 4. Classify trend
        if higher_highs and higher_lows:
            trend = "Strong Uptrend (Bullish)"
            signal=1
        elif lower_highs and lower_lows:
            trend = "Strong Downtrend (Bearish)"
            signal=-1
        elif higher_highs and not higher_lows:
            trend = "Potential Reversal / Broadening Top"
            signal=2
        elif lower_lows and not lower_highs:
            trend = "Potential Reversal / Broadening Bottom"
            signal=-2
        else:
            trend = "Sideways"
            signal=0
            
        # return trend, ph_values, pl_values, signal,
        return trend, signal

    while True:
        rates = mt5.copy_rates_from_pos(symbol,mt5.TIMEFRAME_M1,1,100)
        df=pd.DataFrame(rates)
        # trend, highs, lows, signal = analyze_market_trend(df, left_bars=5, right_bars=5)
        trend, signal= analyze_market_trend(df, left_bars=5, right_bars=5)
        print(trend)
        print('close price: ',df['close'].iloc[-1])
        positions = mt5.positions_get(symbol=symbol)
        
        info_re=info(symbol=symbol)
        if signal==1 and info_re==0:
            print("buy signal")
            buy(symbol=symbol,lot=lot)
        elif signal ==-1 and info_re==0:
            print("sell singal")
            sell(symbol=symbol,lot=lot)
        elif signal==0 or signal==-2 or signal==2:
            close_by_symbol(symbol=symbol,entry=0)

        if positions is None:
            print("No open positions or error fetching positions, error code =", mt5.last_error())
        elif len(positions) > 0:
            print(f"Found {len(positions)} open position(s):")
            for pos in positions:
                print(f"Ticket: {pos.ticket} | Symbol: {pos.symbol} | Type: {'Buy' if pos.type == mt5.POSITION_TYPE_BUY else 'Sell'} | Volume: {pos.volume} | Open Price: {pos.price_open} | Current Profit: {pos.profit}")

                if pos.profit ==0.1:
                    close_by_symbol(symbol=symbol,entry=0) 
                elif pos.profit == -2:
                    close_by_symbol(symbol=symbol,entry=0)
        else:
            print("No open positions currently active.")

        del df
        del signal
        del trend
        del positions
        del rates

        
        

    # --- Example Usage ---
    # trend, highs, lows = analyze_market_trend(df, left_bars=5, right_bars=5)
    # print(f"Trend: {trend}")

    

    
    

except KeyboardInterrupt:

    close_by_symbol(symbol=symbol,entry=0)
    notify_bot("There is a problem with stratagy 6")
