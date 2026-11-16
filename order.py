import MetaTrader5 as mt5
import requests
from datetime import datetime
import joblib
import torch
from torch import nn


real_mode=True
trade_mode=False


def posting_rate_ai(user_title,user_desc,user_source,user_hashtag):
# 1. Define your model architecture
    class EngagementPredictor(nn.Module):

        def __init__(self, title_dim, description_dim, source_dim, hashtag_dim):
            super().__init__()
            self.title_fc = nn.Linear(title_dim, 64)
            self.description_fc = nn.Linear(description_dim, 256)
            self.source_fc = nn.Linear(source_dim, 4)
            self.hashtag_fc = nn.Linear(hashtag_dim, 4)

            total_combined_dim = 64 + 256 + 4 + 4
            self.linear_1 = nn.Linear(total_combined_dim, 64)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.4)
            self.linear_2 = nn.Linear(64, 1)

        def forward(self, title_in, description_in, source_in, hashtag_in):
            t = torch.relu(self.title_fc(title_in))
            d = torch.relu(self.description_fc(description_in))
            s = torch.relu(self.source_fc(source_in))
            h = torch.relu(self.hashtag_fc(hashtag_in))

            combined = torch.cat((t, d, s, h), dim=1)
            x = self.linear_1(combined)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.linear_2(x)
            return x


    # 2. Load your saved vectorizers and model ONCE when the script starts
    print("Loading vectorizers and model...")
    title_vec = joblib.load("title_vectorizer.pkl")
    description_vec = joblib.load("desc_vectorizer.pkl")
    source_vec = joblib.load("source_vectorizer.pkl")
    hashtag_vec = joblib.load("hashtag_vectorizer.pkl")
    # title_vec = joblib.load("news/title_vectorizer.pkl")
    # description_vec = joblib.load("news/desc_vectorizer.pkl")
    # source_vec = joblib.load("news/source_vectorizer.pkl")
    # hashtag_vec = joblib.load("news/hashtag_vectorizer.pkl")

    # Determine dimensions dynamically from the loaded vectorizers
    title_dim = len(title_vec.get_feature_names_out())
    desc_dim = len(description_vec.get_feature_names_out())
    source_dim = len(source_vec.get_feature_names_out())
    hashtag_dim = len(hashtag_vec.get_feature_names_out())

    # Initialize model and load weights
    model = EngagementPredictor(title_dim, desc_dim, source_dim, hashtag_dim)
    model_path = "Post_Rating.pth"
    # model_path = "news/Post_Rating.pth"
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    model.eval()
    print("Ready!")


    # 3. Create a helper function that takes normal strings and handles the rest
    def predict_engagement(
        title_str, description_str, source_str, hashtag_str
    ):
    # Wrap normal strings into lists so vectorizers accept them
        t_input = [title_str]
        d_input = [description_str]
        s_input = [source_str]
        h_input = [hashtag_str]

        # Process and vectorize behind the scenes
        t_feat = title_vec.transform(t_input).toarray()
        d_feat = description_vec.transform(d_input).toarray()
        s_feat = source_vec.transform(s_input).toarray()
        h_feat = hashtag_vec.transform(h_input).toarray()

        # Convert to PyTorch tensors
        t_tensor = torch.from_numpy(t_feat).float()
        d_tensor = torch.from_numpy(d_feat).float()
        s_tensor = torch.from_numpy(s_feat).float()
        h_tensor = torch.from_numpy(h_feat).float()

        # Predict
        with torch.inference_mode():
            prediction = model(t_tensor, d_tensor, s_tensor, h_tensor)

        return prediction.item()


    # ==========================================
    # Example Usage with Normal Strings:
    # ==========================================
    # user_title = input("Title: ")
    # user_desc = input("Description: ")
    # user_source = input("Source: ")
    # user_hashtag = input("Hashtag: ")

    score = predict_engagement(user_title, user_desc, user_source, user_hashtag)
    print(f"\nPredicted Engagement Score: {score:.2f}%")
    return score


def facebook_post(post_text=None, image_url=None):
    if real_mode:
        try:
            """Publishes text, an image, or both to a Facebook Page using the Graph API.
            :param page_id: Your Facebook Page ID
            :param page_access_token: Your Page Access Token
            :param post_text: Optional text message or photo caption
            :param image_url: Optional public URL of the image to post
            """
            page_id ="1361535557047487"
            page_access_token = "EAAVIZCE5Sa6YBSZB3mvtLQPqM16FJduZAJBntCGzrguB4fGrK3sKAXwnV4c7yz1dLLRZB2K1LiudcnByTqHZCdarBWGK1gmZAUn2KaR5IdYy5BEzZAwkEnwFjnR3oDZBYeiRV2DngXjCWdqO1nUrzQiWohOUkfIZBdZChoAPQMseZCzap9Gm83XRY3ZBoZC3Bjw6cZClk888f6"

            # Determine endpoint and payload based on what inputs are provided
            if post_text and image_url:
                # Case: Both text and image are provided
                url = f"https://graph.facebook.com/v26.0/{page_id}/photos"
                payload = {
                    "url": image_url,
                    "caption": post_text,
                    "access_token": page_access_token,
                    # 'published': 'true',     # Explicitly publish to feed
                    # 'no_story': 'false'      # FORCES the post to show up on the timeline feed!
                }
            elif image_url:
                # Case: Only image is provided
                url = f"https://graph.facebook.com/v26.0/{page_id}/photos"
                payload = {"url": image_url, "access_token": page_access_token}
            elif post_text:
                # Case: Only text is provided
                url = f"https://graph.facebook.com/v26.0/{page_id}/feed"
                payload = {"message": post_text, "access_token": page_access_token}
            else:
                print("Error: You must provide at least 'post_text' or an 'image_url'.")
                return None

            try:
                response = requests.post(url, data=payload)
                result = response.json()

                if response.status_code == 200:
                    print(f"Success! Post/Photo ID: {result.get('id') or result.get('post_id')}")
                    return result
                else:
                    print(f"Failed to post: {result}")
                    return result

            except Exception as e:
                print(f"An error occurred: {e}")
                return None


        except Exception as e:
            print("there is a problem: ",e)

    else:
        print("This is in maintanace mode")
            

def binance_post_1(post,image=None):
    if real_mode:
        try:
            # Set your details
            API_KEY = "98cd030b82f6412c9760c8b6cf3aceb2"  # Get this from Binance Creator Center

            # Binance Square Posting Endpoint
            url = "https://www.binance.com/bapi/composite/v1/public/pgc/openApi/content/add"

            headers = {
                "X-Square-OpenAPI-Key": API_KEY,
                "Content-Type": "application/json",
                "clienttype": "binanceSkill"
            }

            payload = {
                "bodyTextOnly": post
            }

            if image is not None:
                response = requests.post(url, json=payload, headers=headers, files=image, timeout=15)

            # Publish the post with a 10-second timeout to prevent hanging
            else:
                response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            # Check HTTP status
            if response.status_code != 200:
                print(f"[Warning] Binance HTTP Error {response.status_code}: {response.text}")
                return False

            data = response.json()

            if data.get("success"):
                post_id = data.get("data", {}).get("id")
                print(f"Published successfully! URL: https://www.binance.com/square/post/{post_id}")
                return True
            else:
                print("[Warning] Binance API failed to publish:", data)
                return False

        except Exception as e:
            # Catches ALL exceptions (network loss, timeouts, bad response formats)
            # Prevents terminal/program from crashing and skips safely
            print(f"[Skipped] Binance post failed due to error: {e}")
            return False
    else:
        print("This is in maintanace mode")


def binance_post_2(post,image=None):
    if real_mode:
        try:
            # Set your details
            API_KEY = "64b585507e874cbdbb3abcb525182c33"  # Get this from Binance Creator Center

            # Binance Square Posting Endpoint
            url = "https://www.binance.com/bapi/composite/v1/public/pgc/openApi/content/add"

            headers = {
                "X-Square-OpenAPI-Key": API_KEY,
                "Content-Type": "application/json",
                "clienttype": "binanceSkill"
            }

            payload = {
                "bodyTextOnly": post
            }

            if image is not None:
                response = requests.post(url, json=payload, headers=headers, files=image, timeout=15)

            # Publish the post with a 10-second timeout to prevent hanging
            else:
                response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            # Check HTTP status
            if response.status_code != 200:
                print(f"[Warning] Binance HTTP Error {response.status_code}: {response.text}")
                return False

            data = response.json()

            if data.get("success"):
                post_id = data.get("data", {}).get("id")
                print(f"Published successfully! URL: https://www.binance.com/square/post/{post_id}")
                return True
            else:
                print("[Warning] Binance API failed to publish:", data)
                return False

        except Exception as e:
            # Catches ALL exceptions (network loss, timeouts, bad response formats)
            # Prevents terminal/program from crashing and skips safely
            print(f"[Skipped] Binance post failed due to error: {e}")
            return False
    else:
        print("This is in maintanace mode")

def notify_bot(text):
    if real_mode:
        requests.post("http://localhost:8080/send", json={"message": text})
    else:
        print("This is in maintanace mode")
        
def modify(tickets,change):

    position=mt5.positions_get(ticket=tickets.order)
    if not position:
        print(f"Position #{tickets} not found or connection lost.")
        return False
    pos = position[0]
    tp=pos.tp
    if change > pos.price_open:
        dif = change - pos.sl          # movement of SL since last time
        if dif > 0:
            tp = pos.tp - dif          # push tp further away by same amount
            # pos.sl = change            # update stored SL for next comparison
            return tp

    # BUY: SL sits below price_open, TP sits above
    elif change < pos.price_open:
        dif = pos.sl - change
        if dif > 0:
            tp = pos.tp + dif
            # pos.sl = change
            return tp
    request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": int(tickets.order),
                "symbol": pos.symbol,
                "sl": change,
                "tp": tp,  # Keeps existing Take Profit value!
                }
    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to update SL. Error: {result.retcode} - {result.comment}")
        # return False
    else:
        # print(f"Position #{tickets} SL successfully updated to {change}")
        print(f"Position SL successfully updated to {change}")
        # return True

def info(symbol):
    positions = mt5.positions_get(symbol=symbol)  # filter by symbol directly

    if not positions:
        print(f"No open positions for {symbol}")
        # print(mt5.positions_get(symbol))
        return 0
    
    for pos in positions:
        print("position open")
        if pos.type==mt5.POSITION_TYPE_BUY:
            print("position type BUY")
            pos_type=1
            return pos_type
        elif pos.type==mt5.POSITION_TYPE_SELL:
            print("position type SELL")
            pos_type=-1
        # pos_type= 1 if pos.type == mt5.POSITION_TYPE_BUY else -1
            return pos_type
    # return 0

def buy(symbol, lot, sl=None,tp=None):
    tick    = mt5.symbol_info_tick(symbol)
    #if want to set sl or tp 1:1 then add this code and make arg sl as input
    # dif=tick.ask-sl
    # tp=tick.ask +dif
    # same=sl
    order = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    symbol,
        "volume":    lot,
        "type":      mt5.ORDER_TYPE_BUY,
        "price":     tick.ask,
      #   "sl":        tick.ask - 100 * point,   # stop loss
      #   "tp":        tick.ask + 100 * point,   # take profit
        "deviation": 10,
        "magic":     234000,
        "comment":   "buy order funciton",
      #   "type_time": mt5.ORDER_TIME_GTC,
      #   "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if tp is not None:
        order["tp"]= tp
    if sl is not None:
        order["sl"]=sl
    result = mt5.order_send(order)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("BUY failed:", result.retcode, result.comment)
    else:
        print("BUY success | Ticket:", result.order)
    return result

def sell(symbol,lot,sl=None,tp=None):
    
    tick = mt5.symbol_info_tick(symbol)
    #if want to set sl or tp 1:1 then add this code and make arg sl as input
    # dif=tick.bid-sl
    # tp=tick.bid +dif
    # same=sl
    order={
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    symbol,
            "volume":    lot,
            "type":      mt5.ORDER_TYPE_SELL,
            "price":     tick.bid,
            # "sl":        sl,  # set loss
            # "tp":        tp,  # take profit
            "deviation": 10,
            "magic":     234000,
            "comment":   "sell order function",
    }
    if tp is not None:
        order["tp"]= tp
    if sl is not None:
        order["sl"]=sl
    result = mt5.order_send(order)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"SELL failed: {result.retcode} {result.comment}")
    else:
        print(f"SELL success | ticket: {result.order} {result.retcode}", )
    return result

def close_by_symbol(symbol,entry):
    positions = mt5.positions_get(symbol=symbol)  # filter by symbol directly
    print(positions)  
    if not positions:
        print(f"No open positions for {symbol}")
        print(mt5.positions_get(symbol))
        enter=0
        return enter

    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        close_type  = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        close_price = tick.bid             if pos.type == 0 else tick.ask
        print(f"pos {pos.symbol}")

        # close_type  = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        #             close_price = tick.bid             if pos.type == 0 else tick.ask
        
        
        if entry==0:
            close_type  = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            close_price = tick.bid             if pos.type == 0 else tick.ask 
        else:
            if entry ==1:
                close_type=mt5.ORDER_TYPE_BUY
                close_price=tick.ask
            else:
                close_price=tick.bid
                close_type=mt5.ORDER_TYPE_SELL


        request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         close_type,
        "position":     pos.ticket,    # still needed internally by MT5
        "price":        close_price,
        "deviation":    10,
        "magic":        234000,
        "comment":      "close by symbol",
        # "type_time":    mt5.ORDER_TIME_GTC,
        # "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        print(f"Closed {pos.symbol} | retcode: {result.retcode}")

def close_all():
    positions = mt5.positions_get()
    
    print("Positions found:", positions)  # ← let's see what it finds
    
    if not positions:
        print("No open positions.")
        return

    for pos in positions:
        tick        = mt5.symbol_info_tick(pos.symbol)
        close_type  = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        close_price = tick.bid             if pos.type == 0 else tick.ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         close_type,
            "position":     pos.ticket,
            "price":        close_price,
            "deviation":    10,
            "magic":        234000,
            "comment":      "close all",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        print(f"Closed {pos.symbol} | retcode: {result.retcode} | {result.comment}")

def loss_manage(trade_limit):
    now=datetime.now()
    today_name = datetime.now().strftime("%A")
    target_time = now.replace(hour=0,minute=1,second=0)
    deals = mt5.history_deals_get(target_time, now)
    if deals is None or len(deals)==0:
        print("No deal found for this day")
        return True
    else:
        total_PL = sum(
            deal.profit + deal.swap + deal.commission
            for deal in deals
            if deal.entry ==mt5.DEAL_ENTRY_OUT
        )
        print(f"Total {total_PL:.2f}")
        # if today_name == "Saturday" or today_name == "Sunday" :
        #     return False
        # if total_PL <= trade_limit or today_name == "Saturday" or today_name == "Sunday" or today_name=="Monday":
        if total_PL<=trade_limit:
            return False
        else:
            return True
      