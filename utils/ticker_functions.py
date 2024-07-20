import os
import numpy as np
import json


BUY_DELISTED_IDX = 0
BUY_LISTED_IDX = 1
BUY_PRICE_MAX_IDX = 2
BUY_PRICE_MIN_IDX = 3
BUY_QUANTITY_MAX_IDX = 4
BUY_QUANTITY_MIN_IDX = 5
BUY_SOLD_IDX = 6
BUY_VALUE_IDX = 7
SELL_DELISTED_IDX = 8
SELL_LISTED_IDX = 9
SELL_PRICE_MAX_IDX = 10
SELL_PRICE_MIN_IDX = 11
SELL_QUANTITY_MAX_IDX = 12
SELL_QUANTITY_MIN_IDX = 13
SELL_SOLD_IDX = 14
SELL_VALUE_IDX = 15

def parse_json_to_clean_tickers(json_object):
    daily_history = []
    for date_string, daily_data in json_object.items():
        ticker = [0 for _ in range(16)]
        ticker[BUY_DELISTED_IDX] = daily_data["buy_delisted"]
        ticker[BUY_LISTED_IDX] = daily_data["buy_listed"]
        ticker[BUY_PRICE_MAX_IDX] = daily_data["buy_price_max"]
        ticker[BUY_PRICE_MIN_IDX] = daily_data["buy_price_min"]
        ticker[BUY_QUANTITY_MAX_IDX] = daily_data["buy_quantity_max"]
        ticker[BUY_QUANTITY_MIN_IDX] = daily_data["buy_quantity_min"]
        ticker[BUY_SOLD_IDX] = daily_data["buy_sold"]
        ticker[BUY_VALUE_IDX] = daily_data["buy_value"]
        ticker[SELL_DELISTED_IDX] = daily_data["sell_delisted"]
        ticker[SELL_LISTED_IDX] = daily_data["sell_listed"]
        ticker[SELL_PRICE_MAX_IDX] = daily_data["sell_price_max"]
        ticker[SELL_PRICE_MIN_IDX] = daily_data["sell_price_min"]
        ticker[SELL_QUANTITY_MAX_IDX] = daily_data["sell_quantity_max"]
        ticker[SELL_QUANTITY_MIN_IDX] = daily_data["sell_quantity_min"]
        ticker[SELL_SOLD_IDX] = daily_data["sell_sold"]
        ticker[SELL_VALUE_IDX] = daily_data["sell_value"]
        daily_history.append(ticker)
    return daily_history


def parse_and_clean_all_tickers(folder_path, save_path):
    print("Parsing JSON data...")
    for file_name in os.listdir(folder_path):
        if ".json" not in file_name:
            continue
        print("Parsing",file_name)
        with open(os.path.join(folder_path, file_name),'r') as f:
            json_object = dict(json.load(f))

        clean_ticker_history = parse_json_to_clean_tickers(json_object)
        clean_ticker_history = np.asarray(clean_ticker_history, dtype=np.float32)
        can_save = validate_ticker(clean_ticker_history)

        if can_save:
            remove_strange_outliers(clean_ticker_history)
            full_save_path = os.path.join(save_path, "{}".format(file_name.replace(".json", "")))
            np.save(full_save_path, clean_ticker_history)
    print("All JSON data parsed!")

def validate_ticker(ticker_history):
    if len(ticker_history) < 365:
        return False

    if np.isnan(ticker_history).any():
        return False

    n_sold = 0
    n_bought = 0
    for i in range(len(ticker_history)):
        n_sold += ticker_history[i][SELL_SOLD_IDX]
        n_bought += ticker_history[i][BUY_SOLD_IDX]

    if n_sold < 2000 or n_bought < 2000:
        return False

    return True


def remove_strange_outliers(ticker_history):
    sell_price_history = ticker_history[:, SELL_PRICE_MIN_IDX]

    start_point = 1
    for i in range(1, len(sell_price_history)):
        if sell_price_history[i] >= sell_price_history[i - 1] * 2:
            start_point = i
            break
    end_point = start_point + 10

    indices = [SELL_PRICE_MIN_IDX, SELL_PRICE_MAX_IDX, SELL_QUANTITY_MAX_IDX, SELL_QUANTITY_MIN_IDX, SELL_LISTED_IDX,
               SELL_DELISTED_IDX,
               BUY_PRICE_MIN_IDX, BUY_PRICE_MAX_IDX, BUY_QUANTITY_MIN_IDX, BUY_QUANTITY_MAX_IDX, BUY_LISTED_IDX,
               BUY_DELISTED_IDX]

    for idx_of_interest in indices:
        slice_of_interest = ticker_history[:, idx_of_interest]

        if slice_of_interest[start_point - 1] == 0:
            divisor = 1
        else:
            divisor = slice_of_interest[start_point - 1]

        error_region_norm_const = slice_of_interest[start_point] / divisor
        if error_region_norm_const == 0:
            error_region_norm_const = 2

        if idx_of_interest in [11, 3, 5, 13]:
            ticker_history[start_point:end_point - 1, idx_of_interest] /= error_region_norm_const
        else:
            ticker_history[start_point:end_point, idx_of_interest] /= error_region_norm_const


def get_item_ids_from_names(item_names):
    print("Mapping item names to IDs...")
    item_ids = []
    with open("H:/programming/gw2 tp data/datawars2/api-datawars2-ie_gw2_v1_items.csv",'r') as ticker_csv:
        lines = ticker_csv.readlines()
        for line in lines[1:]:
            splt = line.split(",")
            item_id = int(splt[0])
            item_name = splt[1]
            for i in range(len(item_names)):
                if item_names[i] == item_name:
                    item_ids.append(item_id)
                    print(item_names[i], "->", item_id)
                    break

            if len(item_names) == len(item_ids):
                break
    return item_ids

def compute_data_stats(folder_path):
    from utils import WelfordRunningStat
    stats = WelfordRunningStat(16)
    for file_name in os.listdir(folder_path):
        data = np.load(os.path.join(folder_path, file_name))
        stats.increment(data, len(data))

    mean = "["
    for arg in stats.mean:
        mean = "{},{}".format(mean, arg)
    mean = "{}]".format(mean)

    std = "["
    for arg in stats.std:
        std = "{},{}".format(std, arg)
    std = "{}]".format(std)

    print(mean, " | ", std)


def get_known_ticker_stats():
    mean = np.asarray(
        [171.42047119140625, 827.7025756835938, 11676.0439453125, 11326.80859375, 18964.408203125, 18426.13671875,
         646.0158081054688, 271612.84375, 1057.72607421875, 5196.5078125, 15439.193359375, 14849.068359375,
         71970.765625, 70869.578125, 3590.345703125, 559982.3125])
    std = np.asarray(
        [51037.878402263545, 90707.43915931908, 98491.31129172258, 92930.96308634736, 172001.61742100184,
         166843.26221175422, 36576.04212160986, 72555621.569429, 34273.580322012334, 310141.650594909,
         119509.74761516842, 112436.68001417398, 745242.3897950264, 727350.9994668862, 294474.04134367383,
         25245300.946669985])

    return mean, std


def graph_train_data():
    from training.supervised.data_management import data_loader
    from utils import graphing_helper as graph
    tickers = [62980,8910,26832,45986,20741,24597,9423,13076,458,19763]
    tickers = [9423]
    tickers = ["{}.npy".format(arg) for arg in tickers]

    mean, std = get_known_ticker_stats()
    n_tickers_to_load = 10
    data, stats, shortest_length = data_loader.load_specific_tickers("H:/programming/gw2 tp data/datawars2/clean_v1_tickers", tickers)
    legend = []
    idx = SELL_PRICE_MIN_IDX
    alpha = 0#0.95
    for key, value in data.items():
        price_history = []
        ema = None
        for ticker_day in value:
            # standardized_data = np.subtract(ticker_day, mean) / std
            standardized_data = ticker_day
            if ema is None:
                ema = standardized_data[idx] / 10000
            else:
                ema = ema*alpha + standardized_data[idx]*(1 - alpha) / 10000

            price_history.append(ema)

        graph.plot_data(price_history, clear=False)
        legend.append(key)
    graph.set_legend(legend)
    graph.save_plot("train_data_debug")


def process_ticker_to_train_data(ticker_data, days_to_wait_for_sale=14):
    x = []
    y = []
    print(np.shape(ticker_data))
    mean, std = get_known_ticker_stats()

    for i in range(len(ticker_data)-days_to_wait_for_sale):
        standardized_ticker = np.subtract(ticker_data[i], mean) / std
        x.append(standardized_ticker)
        can_buy = False
        can_sell = False
        buy_bid = ticker_data[i][BUY_PRICE_MAX_IDX] + 1
        sell_bid = None

        # if there are no listings for sale don't try to buy one
        if buy_bid == 1:
            y.append(-1)
            continue

        #1. submit a buy order immediately by over-cutting the highest bidder by 1
        #2. wait at most 14 days to see if the item has been sold within that time at or below the bid price
        #3. if the item has sold, we own one, so we undercut the lowest sale offer by 1 immediately
        #4. if in the remaining time to wait the item will sell at our offered price, the flip was successful
        for days_waited in range(1, days_to_wait_for_sale):
            current_day = ticker_data[i + days_waited]
            item_can_be_bought = current_day[BUY_SOLD_IDX] > 0 and current_day[BUY_PRICE_MAX_IDX] <= buy_bid
            if item_can_be_bought and not can_buy:
                can_buy = True
                sell_bid = current_day[SELL_PRICE_MIN_IDX] - 1

            if sell_bid is not None and can_buy:
                item_can_be_sold = current_day[SELL_SOLD_IDX] > 0 and current_day[SELL_PRICE_MIN_IDX] >= sell_bid
                if item_can_be_sold and not can_sell:
                    can_sell = True

            if can_buy and can_sell:
                break

        # if this item can be flipped label it with the percent change in money we earned by flipping it
        if can_buy and can_sell:
            profit = sell_bid*0.85 - buy_bid
            percent_gain = profit / buy_bid
            y.append(percent_gain)

        # if this item can't be flipped label it as a complete loss
        else:
            y.append(-1)

    return np.asarray(x), np.asarray(y).reshape(-1, 1)


def create_training_data(parsed_data_path, training_data_save_path, val_data_save_path, val_data_item_names, days_to_wait_for_sale=14):
    print("Creating training & validation data from processed JSONs...")
    from training.supervised.data_management import data_loader
    train_x_path = os.path.join(training_data_save_path, "inputs")
    train_y_path = os.path.join(training_data_save_path, "labels")
    val_x_path = os.path.join(val_data_save_path, "inputs")
    val_y_path = os.path.join(val_data_save_path, "labels")

    os.makedirs(train_x_path, exist_ok=True)
    os.makedirs(train_y_path, exist_ok=True)
    os.makedirs(val_x_path, exist_ok=True)
    os.makedirs(val_y_path, exist_ok=True)

    val_item_ids = get_item_ids_from_names(val_data_item_names)

    for file_name in os.listdir(parsed_data_path):
        print("Processing {}...".format(file_name))
        ticker_data, _, __ = data_loader.load_specific_tickers(parsed_data_path, [file_name], raw_data=True, smooth=True)
        x, y = process_ticker_to_train_data(list(ticker_data.values())[0], days_to_wait_for_sale=days_to_wait_for_sale)

        data_name = file_name[:file_name.rfind(".")]
        if int(data_name) in val_item_ids:
            x_path = val_x_path
            y_path = val_y_path
        else:
            x_path = train_x_path
            y_path = train_y_path

        np.save(os.path.join(x_path, data_name), x)
        np.save(os.path.join(y_path, data_name), y)


def smooth_tickers(ticker_data, alpha=0.9):
    print("SMOOTHING TICKER DATA")
    smoothed_data = {}
    one_minus_alpha = 1 - alpha

    for ticker_id, ticker_history in ticker_data.items():
        emas = [arg for arg in ticker_history[0]]
        smoothed_data[ticker_id] = []
        for day in ticker_history:
            for i in range(len(day)):
                emas[i] = emas[i]*alpha + day[i]*one_minus_alpha
            smoothed_data[ticker_id].append(np.asarray([arg for arg in emas]))

    return smoothed_data