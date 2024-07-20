import numpy as np
from training.supervised.data_management import DataProcessor
import gym
from gym.spaces import MultiDiscrete, Box
from utils import ticker_functions

class TradingEnvironment(gym.Env):
    ACTION_MAP = {0: "submit_buy_listing", 1: "submit_sell_listing", 2: "cancel_buy_listing", 3: "cancel_sell_listing", 4: "no-op"}
    N_TICKERS = 10
    SEQ_LEN = 90

    def __init__(self):
        n_tickers = TradingEnvironment.N_TICKERS
        self.rng = np.random.RandomState(123)
        self.current_sequence = None
        self.prev_slice = None
        self.sequence_idx = 0
        self.n_actions_per_item = len(list(TradingEnvironment.ACTION_MAP.keys()))

        self.growing_sell_listing_history = [[] for _ in range(n_tickers)]

        # (number of units currently listed, price asked, days since listing)
        self.current_sell_listings = [[0, 0, 0] for _ in range(n_tickers)]

        # (number of units currently attempting to buy, price offered, days since listing)
        self.current_buy_listings = [[0, 0, 0] for _ in range(n_tickers)]

        # number of units owned
        self.current_units_owned = [0 for _ in range(n_tickers)]

        self.prev_action = np.asarray([0 for _ in range(self.n_actions_per_item)])

        self.current_copper = 0

        self.obs_size = DataProcessor.TICKER_LENGTH*n_tickers + \
                        len(self.current_sell_listings) + \
                        len(self.current_buy_listings) + \
                        len(self.current_units_owned) + \
                        len(self.prev_action) + 1

        self.data_processor = DataProcessor(n_tickers, TradingEnvironment.SEQ_LEN, 1, 1, n_processes=1)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_size,))
        self.action_space = MultiDiscrete(nvec=[self.n_actions_per_item for _ in range(n_tickers)], seed=123)
        self.mean, self.std = ticker_functions.get_known_ticker_stats()


    def reset(self, *args, **kwargs):
        n_tickers = TradingEnvironment.N_TICKERS
        self.current_copper = 1000*10000 # 1,000 gold
        self.sequence_idx = 0

        x, y, q = self.data_processor.get_random_batch()
        self.current_sequence = x[0]

        self.prev_slice = np.zeros_like(self.current_sequence[0])

        self.growing_sell_listing_history = [[] for _ in range(n_tickers)]
        self.current_sell_listings = [[0, 0, 0] for _ in range(n_tickers)]
        self.current_buy_listings = [[0, 0, 0] for _ in range(n_tickers)]
        self.current_units_owned = [0 for _ in range(n_tickers)]
        self.prev_action = [0 for _ in range(n_tickers)]

        return self._build_obs()

    def step(self, action):
        prev_copper = self.current_copper
        self._check_market()
        self._take_action(action)
        reward = (self.current_copper - prev_copper) / 10000

        obs = self._build_obs()
        
        for i in range(len(self.current_buy_listings)):
            # increment days since listings by 1 and reset necessary entries
            if self.current_buy_listings[i][0] > 0:
                self.current_buy_listings[i][2] += 1
            else:
                self.current_buy_listings[i] = [0, 0, 0]

            if self.current_sell_listings[i][0] > 0:
                self.current_sell_listings[i][2] += 1
            else:
                self.current_sell_listings[i] = [0, 0, 0]
                
        self.prev_action[:] = action[:]
        self.prev_slice = self.current_sequence[self.sequence_idx]
        self.sequence_idx += 1

        done = self.sequence_idx >= TradingEnvironment.SEQ_LEN or self.current_copper <= 0

        return obs, reward, done, {}

    def _check_market(self):
        """
        Check if listed items have sold or buy orders have come through, update money and item holdings accordingly.

        :return: None
        """
        current_slice = self.current_sequence[self.sequence_idx]
        prev_slice = self.prev_slice
        n_tickers = TradingEnvironment.N_TICKERS
        ticker_length = DataProcessor.TICKER_LENGTH
        for i in range(n_tickers):
            start = i*ticker_length
            stop = start + ticker_length
            current_day_ticker = current_slice[start:stop]

            current_instant_buy_price = current_day_ticker[ticker_functions.SELL_PRICE_MIN_IDX]
            current_instant_sell_price = current_day_ticker[ticker_functions.BUY_PRICE_MAX_IDX]

            n_sell_listings = self.current_sell_listings[i][0]
            n_buy_offers = self.current_buy_listings[i][0]

            if n_sell_listings > 0:
                if current_instant_buy_price > self.current_sell_listings[i][1]:
                    # 10% tax on all sales
                    self.current_copper += self.current_sell_listings[i][1] * 0.9
                    self.current_sell_listings[i] = [0, 0, 0]

            if n_buy_offers > 0:
                price_offered = self.current_buy_listings[i][1]
                if current_instant_sell_price < price_offered:
                    self.current_buy_listings[i][0] = 0
                    self.current_units_owned[i] += n_buy_offers

    def _take_action(self, actions):
        """
        Apply one of ACTION_MAP actions to each item in the group of tickers at the current day. Update money and item
        holdings accordingly.

        :param actions: Actions to apply.
        :return: None.
        """

        current_slice = self.current_sequence[self.sequence_idx]
        amap = TradingEnvironment.ACTION_MAP
        n_tickers = TradingEnvironment.N_TICKERS
        ticker_length = DataProcessor.TICKER_LENGTH

        for i in range(n_tickers):
            start = i * ticker_length
            stop = start + ticker_length
            current_day_ticker = current_slice[start:stop]
            act = amap[actions[i]]
            if act == "submit_buy_listing":
                if self.current_buy_listings[i][0] == 0:
                    instant_sell_price = current_day_ticker[ticker_functions.BUY_PRICE_MAX_IDX]
                    price_to_offer = instant_sell_price + 1 # overcut by 1 copper
                else:
                    price_to_offer = self.current_buy_listings[i][1] # force model to only submit buy orders at once price for simplicity
                self.current_copper -= price_to_offer
                self.current_buy_listings[i][0] += 1
                self.current_buy_listings[i][1] = price_to_offer
                self.current_buy_listings[i][2] = 0
                
            elif act == "submit_sell_listing":
                if self.current_units_owned[i] > 0:
                    if self.current_sell_listings[i][0] > 0:
                        price_to_ask = self.current_sell_listings[i][1]
                    else:
                        instant_buy_price = current_day_ticker[ticker_functions.SELL_PRICE_MIN_IDX]
                        price_to_ask = instant_buy_price - 1  # undercut by 1 copper
                    
                    self.current_sell_listings[i][0] += 1
                    self.current_sell_listings[i][1] = price_to_ask
                    self.current_sell_listings[i][2] = 0
                    self.current_units_owned[i] -= 1
                    self.current_copper -= price_to_ask*0.05 # 5% listing fee

            elif act == "cancel_buy_listing":
                if self.current_buy_listings[i][0] > 0:
                    self.current_copper += self.current_buy_listings[i][1]
                    self.current_buy_listings[i][0] -= 1
                    if self.current_buy_listings[i][0] <= 0:
                        self.current_buy_listings[i] = [0, 0, 0]

            elif act == "cancel_sell_listing":
                if self.current_units_owned[i] > 0:
                    self.current_sell_listings[i][0] -= 1
                    self.current_units_owned[i] += 1
                    if self.current_sell_listings[i][0] <= 0:
                        self.current_sell_listings[i] = [0,0,0]

            else:
                continue

    def _build_obs(self):
        current_slice = self.current_sequence[self.sequence_idx]
        obs = []
        obs += current_slice
        obs += [arg for sublist in self.current_sell_listings for arg in sublist]
        obs += [arg for arg in self.current_units_owned]
        obs += [arg for sublist in self.current_buy_listings for arg in sublist]
        obs += [arg for arg in np.ravel(self.prev_action)]
        obs.append(self.current_copper)

        return np.asarray(obs)

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
        self.action_space = MultiDiscrete(nvec=[self.n_actions_per_item for _ in range(TradingEnvironment.N_TICKERS)], seed=123)

    def render(self, mode="human"):
        if self.sequence_idx >= len(self.current_sequence):
            return

        current_slice = self.current_sequence[self.sequence_idx]
        n_tickers = TradingEnvironment.N_TICKERS
        ticker_length = DataProcessor.TICKER_LENGTH

        print("-"*10)
        for i in range(n_tickers):
            start = i * ticker_length
            stop = start + ticker_length
            current_day_ticker = current_slice[start:stop]
            current_instant_buy_price = current_day_ticker[ticker_functions.SELL_PRICE_MIN_IDX]
            current_instant_sell_price = current_day_ticker[ticker_functions.BUY_PRICE_MAX_IDX]
            print("TICKER: {}\n"
                  "INSTANT BUY PRICE: {:7.6f}\n"
                  "INSTANT SELL PRICE: {:7.6f}\n"
                  "CURRENT HOLDINGS: {}\n"
                  "BUY LISTINGS: {} | {:7.6f} | {}\n"
                  "SELL LISTINGS: {} | {:7.6f} | {}\n".format(
                i,
                current_instant_buy_price/10000,
                current_instant_sell_price/10000,
                self.current_units_owned[i],
                self.current_buy_listings[i][0], self.current_buy_listings[i][1]/10000, self.current_buy_listings[i][2],
                self.current_sell_listings[i][0], self.current_sell_listings[i][1]/10000,  self.current_sell_listings[i][2]
            ))
        print("CURRENT GOLD: {:7.4f}".format(self.current_copper/10000))
        print("-"*10)
        print()

    def close(self):
        self.data_processor.cleanup()

