import numpy as np
from training.reinforcement import TradingEnvironment
from utils import ticker_functions

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def main():
    print("creating env")
    env = TradingEnvironment()
    n_tickers = TradingEnvironment.N_TICKERS
    ticker_length = env.data_processor.TICKER_LENGTH
    print("resetting env")
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        print("stepping env")
        start = 0
        slice_size = ticker_length*n_tickers
        current_slice = obs[start:start+slice_size]

        start += slice_size
        sell_listing_size = 3*n_tickers
        current_sell_listings = obs[start:start+sell_listing_size]

        start += sell_listing_size
        owned_units_size = n_tickers
        n_items_owned = obs[start:start+owned_units_size]

        start += owned_units_size
        buy_listing_size = 3*n_tickers

        current_buy_listings = obs[start:start+buy_listing_size]

        current_copper = obs[-1]

        print(len(current_sell_listings), " | ",len(current_buy_listings), len(obs))
        actions = [4 for _ in range(n_tickers)]
        print("looping over tickers")
        for i in range(n_tickers):
            start = i*ticker_length
            stop = start + ticker_length
            ticker = current_slice[start:stop]
            instant_buy_price = ticker[ticker_functions.SELL_PRICE_MIN_IDX]
            instant_sell_price = ticker[ticker_functions.BUY_PRICE_MAX_IDX]

            profit = instant_buy_price*0.85 - instant_sell_price

            if profit > 0 and instant_sell_price < current_copper:
                current_copper -= instant_sell_price
                actions[i] = 0 # submit buy offer
            elif n_items_owned[i] > 0:
                actions[i] = 1 # submit sell listing
            elif current_sell_listings[i*3] > 0 and current_sell_listings[i*3+2] > 10:
                actions[i] = 3 # cancel sell listing
            elif current_buy_listings[i*3] > 0 and current_buy_listings[i*3+2] > 1:
                actions[i] = 2 # cancel buy listing


        print("submitting actions")
        obs, rew, done, _ = env.step(actions)
        total_reward += rew

        print("rendering env")
        env.render()

    env.close()

if __name__ == "__main__":
    main()