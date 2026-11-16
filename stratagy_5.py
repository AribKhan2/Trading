import time
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from stratagy.order import buy,sell,loss_manage,info,close_by_symbol,notify_bot
import threading



# Initialize MT5 safely
if not mt5.initialize():
    print(f"MetaTrader initialization failed: {mt5.last_error()}")
    quit()
else:
    print("MetaTrader connected successfully.")

# symbol = "GBPUSDm"

def grid_bounds(symbol: str, num_grids: int = 20):
    """
    Fetches 500 Daily candles (D1) and extracts the highest high 
    and lowest low across the full 500-bar history.
    """
    # Fetch last 500 completed daily candles (skipping bar 0)
    rates_1D = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 1, 1000)
    if rates_1D is None or len(rates_1D) == 0:
        return None, None
        
    df_1D = pd.DataFrame(rates_1D)
    
    # 1. Highest high & lowest low across ALL 500 bars
    upper_price = df_1D['high'].max()
    lower_price = df_1D['low'].min()
    
    # 2. Arithmetic Grid array using NumPy linspace
    grid = np.linspace(lower_price, upper_price, num_grids + 1)
    
    return grid, lower_price, upper_price

def evaluate_grid_signal(grid: np.ndarray, current_price: float):
    """
    Evaluates where current price is inside the 500-day grid.
    Returns: (closest_grid_index, trade_type, entry_price, tp_price, sl_price)
    """
    # Vectorized absolute distance to find closest grid level index
    distances = np.abs(grid - current_price)
    i = int(np.argmin(distances)) # Grid index (0 to num_grids)
    
    # Median price of the entire 500-day range
    grid_midpoint = np.median(grid)
    
    # Check boundaries to avoid array index out of bounds
    if i == 0 or i == len(grid) - 1:
        return None  # Price at extreme boundary
        
    level_price = grid[i]
    
    # --- BINANCE-STYLE DIRECTION BIAS ---
    # Discount Zone (Price below 500-day median) -> BUY Bias
    if current_price < grid_midpoint:
        trade_type = "BUY"
        entry_price = level_price
        tp_price = grid[i + 1]  # Take Profit = Next level higher (i + 1)
        sl_price = grid[i - 1]  # Stop Loss = Next level lower (i - 1)
        
    # Premium Zone (Price above 500-day median) -> SELL Bias
    else:
        trade_type = "SELL"
        entry_price = level_price
        tp_price = grid[i - 1]  # Take Profit = Next level lower (i - 1)
        sl_price = grid[i + 1]  # Stop Loss = Next level higher (i + 1)
        
    return {
        "index": i,
        "type": trade_type,
        "entry": entry_price,
        "tp": tp_price,
        "sl": sl_price
    }


# Main Execution Loop
try:
    def main(symbol,lot,num_grid,canddle):
        while True:
            grid, lower, upper =grid_bounds(symbol=symbol,num_grids=num_grid)
            rate = mt5.copy_rates_from_pos(symbol,mt5.TIMEFRAME_M5,0,canddle)
            info_re=info(symbol=symbol)
            signal = evaluate_grid_signal(grid=grid,current_price=float(rate['close'][-1]))

            print("type: ",signal['type'])
            print("entry: ",signal["entry"])
            print("tp: ",signal["tp"])
            print("sl: ",signal["sl"])  
            if info_re==0 and signal['type'] == "BUY":
                buy(symbol=symbol,lot=lot,tp=signal['tp'],sl=signal['sl'])
            elif info_re==0 and signal['type'] == "SELL":
                sell(symbol=symbol,lot=lot,tp=signal['tp'],sl=signal['sl'])
            else: 
                print("No action is taken")

            time.sleep(5)
        

except KeyboardInterrupt:
    print("Grid loop stopped by user.")
    mt5.shutdown()


# 1. Create individual thread objects
t1 = threading.Thread(target=main, args=("GBPUSDm",0.01,40,1000))
t2 = threading.Thread(target=main, args=("EURUSDm",0.01,40,1000))
t3 = threading.Thread(target=main, args=("EURGBPm",0.01,40,1000))

start_time = time.time()

# 2. Start the threads
t1.start()
t2.start()
t3.start()

# 3. Wait for both threads to finish before moving on
# t1.join()
# t2.join()

end_time = time.time()