import pandas as pd
import numpy as np
import math


def ema_mt_style(close: pd.Series, length: int) -> pd.Series:
        """
        Exponential Moving Average, seeded the way MetaTrader (and hence most
        MetaTrader-based brokers, incl. Exness) does it:
    
        1. The first `length` bars are warm-up: no value is emitted (NaN).
        2. EMA[length-1] = SMA(close[0:length])          <- SMA seed
        3. EMA[i] = alpha * close[i] + (1 - alpha) * EMA[i-1]   for i >= length
            where alpha = 2 / (length + 1)
    
        The recursive part (step 3) is handed off to pandas' C-implemented
        `.ewm()` for speed — only the seed itself is computed by hand.
        """
        if length < 1:
            raise ValueError("length must be >= 1")
        if isinstance(close, pd.DataFrame) or getattr(close, "ndim", 1) != 1:
            raise TypeError(
                "close must be a 1-D pandas Series, got something "
                f"{getattr(close, 'ndim', '?')}-dimensional instead. "
                "If you selected a column with df[['close']] (double brackets) "
                "or your df has duplicate column names, use df['close'].squeeze() "
                "or drop the duplicate columns first."
            )
    
        alpha = 2 / (length + 1)
        ema = pd.Series(np.nan, index=close.index, dtype=float)
    
        n = len(close)
        if n < length:
            return ema
    
        seed = close.iloc[:length].mean()
    
        # Do the recursion in plain numpy so index alignment/labels can never
        # cause a crash here, no matter what index `close` has.
        vals = close.iloc[length - 1:].to_numpy(dtype=float, copy=True)
        vals[0] = seed
        tail_ema = pd.Series(vals).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    
        ema.iloc[length - 1:] = tail_ema
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
        
        # Add EMA length for the initial SMA seed period
        total_bars = math.ceil(required_y) + ema_length
        return total_bars
def get_bollinger_bands(close: pd.Series,std_dev:float,period: int)-> dict:
    """
    Fetches market data and returns current price, BB levels, and signal direction.
    Signals: +1 (BUY / Oversold rebound), -1 (SELL / Overbought drop), 0 (NEUTRAL)
    """
    # 2. Convert to DataFrame
    df = close

    # 3. Calculate Bollinger Bands using rolling math
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    df_std = df['close'].rolling(window=period).std()

    df['bb_upper'] = df['bb_middle'] + (std_dev * df_std)
    df['bb_lower'] = df['bb_middle'] - (std_dev * df_std)

    # 4. Extract latest closed candle values (Index -2 ensures closed bar execution)
    latest = df.iloc[-1]
    current_close = latest['close']
    upper_band = latest['bb_upper']
    lower_band = latest['bb_lower']
    middle_band = latest['bb_middle']

    # 5. Determine Trading Signal
    signal = 0
    print(current_close > upper_band)
    print(current_close <=upper_band)
    print(current_close >= lower_band)
    print(current_close < lower_band )
    # Pack results into clean dictionary
    bb_data = {
        "close": current_close,
        "upper": round(upper_band, 5),
        "middle": round(middle_band, 5),
        "lower": round(lower_band, 5),
        "signal": signal
    }

    # Explicit memory cleanup to keep RAM flat
    del df, rates

    return bb_data
def analyze_n_bar_trend_sr(df: pd.DataFrame, n_lookback: int, m_forecast: int = 3) -> pd.DataFrame:
    """
    Calculates current N-bar trend state, dynamic Support/Resistance levels over N bars,
    and projects trend state transition M bars into the future.
    """
    df = df.copy()
    
    # 1. Baseline Price: Close from N bars ago
    df['close_n_ago'] = df['close'].shift(n_lookback)
    
    # 2. Dynamic Support & Resistance over the N-bar window (including current bar)
    df['resistance_level'] = df['high'].rolling(window=n_lookback + 1).max()
    df['support_level']    = df['low'].rolling(window=n_lookback + 1).min()
    
    # 3. Current Trend State (Pure observation, no signals)
    df['current_state'] = 'NEUTRAL'
    df.loc[df['close'] > df['close_n_ago'], 'current_state'] = 'UPTREND'
    df.loc[df['close'] < df['close_n_ago'], 'current_state'] = 'DOWNTREND'
    
    # 4. Per-Bar Linear Momentum Slope over N bars
    df['per_bar_momentum'] = (df['close'] - df['close_n_ago']) / n_lookback
    
    # 5. Project Price M bars ahead
    df['projected_price_m_ahead'] = df['close'] + (df['per_bar_momentum'] * m_forecast)
    
    # 6. Forecasted State M bars into the future
    df['forecasted_state_m_ahead'] = 'NEUTRAL'
    df.loc[df['projected_price_m_ahead'] > df['close_n_ago'], 'forecasted_state_m_ahead'] = 'UPTREND'
    df.loc[df['projected_price_m_ahead'] < df['close_n_ago'], 'forecasted_state_m_ahead'] = 'DOWNTREND'
    
    # 7. Expected State Transition
    df['expected_transition'] = 'STABLE'
    df.loc[
        (df['current_state'] != df['forecasted_state_m_ahead']), 
        'expected_transition'
    ] = 'REVERSAL / SHIFT'
    
    # Cleanup baseline column
    df.drop(columns=['close_n_ago'], inplace=True)
    
    return df
def calculate_supertrend_raw(df: pd.DataFrame, length:int, multiplier:float)->pd.DataFrame:
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
    df['trend_dir'] = direction
    
    # Continuous indicator signal mapping
    df['signal'] = df['trend_dir']  # 1 for Green line, -1 for Red line
    
    return df
def calculate_macd_pandas(df:pd.DataFrame, fast=12, slow=26, signal=9, price_col='close'):
    """
    Calculates MACD Line, Signal Line, and Histogram using Pandas.
    It is Incomplete So try not to use
    """
    df = df.copy()
    
    # 1. Calculate Fast and Slow EMAs
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    
    # 2. MACD Line = Fast EMA - Slow EMA
    df['macd_line'] = ema_fast - ema_slow
    
    # 3. Signal Line = EMA of MACD Line
    df['signal_line'] = df['macd_line'].ewm(span=signal, adjust=False).mean()
    
    # 4. Histogram = MACD Line - Signal Line
    df['macd_hist'] = df['macd_line'] - df['signal_line']
    
    return df
def calculate_fibonacci_levels(df:pd.DataFrame)->dict:
    """
    Extracts the latest active Swing High and Low and generates Fib retracement levels.
    It Give Fibo: 0.0,23.6,38.2,50.0,61.8,78.6,100
    """
    # Grab the most recent valid Swing High and Low values
    # latest_high = df['Swing_High'].dropna().iloc[-1]
    # latest_low = df['Swing_Low'].dropna().iloc[-1]
    latest_high= df["high"].max()
    latest_low= df["low"].min()
    # Determine the distance (price range)
    price_diff = latest_high - latest_low
    
    # Calculate Fibonacci Retracement Levels
    fib_levels = {
        '0.0% (Low)':   latest_low,
        '23.6%':        latest_low + (price_diff * 0.236),
        '38.2%':        latest_low + (price_diff * 0.382),
        '50.0%':        latest_low + (price_diff * 0.500),
        '61.8%':        latest_low + (price_diff * 0.618),
        '78.6%':        latest_low + (price_diff * 0.786),
        '100.0% (High)': latest_high
    }
    
    return fib_levels, latest_high, latest_low
def find_swing_points(df, window=39):
        
    # Finds local highs and lows over a specified lookback window.
    # A High is a Swing High if it is the maximum within 'window' bars left & right.
    # A Low is a Swing Low if it is the minimum within 'window' bars left & right.

    # df = df.copy()

    # Identify local peaks and troughs
    df['Swing_High'] = df['high'][(df['high'] == df['high'].rolling(center=True, window=2*window+1).max())]
    df['Swing_Low'] = df['low'][(df['low'] == df['low'].rolling(center=True, window=2*window+1).min())]

    return df
def calculate_rsi_numpy(close: np.ndarray, length: int = 14) -> np.ndarray:
    """
    Calculates Relative Strength Index (RSI) using Wilder's Smoothing via NumPy.
    Matches TradingView / MT5 standard calculations exactly.
    
    :param prices: 1D NumPy array of close prices
    :param length: Lookback period (default: 14)
    :return: 1D NumPy array with calculated RSI values
    """
    prices=close
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



