from gw2spidy import Gw2Spidy
import numpy as np
import time
import json
import os
from datetime import datetime
from utils import graphing_helper as graph
from utils import ticker_functions
from training.supervised import Learner, Validator
from training.supervised.data_management import DataProcessor


def get_timestamp(datetime_string):
    datetimeObj = datetime.strptime(datetime_string[:-5], '%Y-%m-%dT%H:%M:%S')
    return datetimeObj.timestamp()


def train():
    learner = Learner()
    try:
        learner.learn()
    except:
        import traceback
        print("EXCEPTION\n",traceback.format_exc())
    finally:
        learner.data_processor.cleanup()


def validate():
    val = Validator(seq_len=Learner.SEQ_LEN + Learner.TRADE_DURATION, ticker_length=DataProcessor.TICKER_LENGTH, n_tickers=Learner.N_TICKERS)
    val.test_model(n_training_data_tests=1000)


def download_more_detailed_tp_data(save_path):
    import requests
    # base_url = "https://api.datawars2.ie/gw2/v2/history/hourly/json?itemID="
    base_url = "https://api.datawars2.ie/gw2/v1/history?itemID="
    item_ids = []
    existing_item_ids = []
    for file_name in os.listdir(save_path):
        existing_item_ids.append(file_name.replace(".json", ""))
    with open("H:/programming/gw2 tp data/datawars2/api-datawars2-ie_gw2_v1_items.csv", 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            splt = line.split(",")
            item_id = int(splt[0])
            if str(item_id) not in existing_item_ids:
                item_ids.append(item_id)

    for item_id in item_ids:
        url = "{}{}".format(base_url, item_id)
        print("downloading",url,"...")
        response = requests.get(url)
        item_data = response.json()
        if not item_data:
            print("unable to find history for item ID {}!".format(item_id))
            continue

        print(item_data)
        full_history_json = {}
        for json_object in item_data:
            if 'buy_sold' not in json_object.keys():
                continue
            date = json_object['date']
            json_object['date'] = get_timestamp(date)
            full_history_json[date] = json_object
        file_path = os.path.join(save_path, "{}.json".format(item_id))
        with open(file_path, 'w') as f:
            f.write(json.dumps(full_history_json, indent=4))


def debug_ticker():
    # desired_items = ["Mystic Coin", "Glob of Ectoplasm", "Bolt of Damask"]
    desired_items = ["Stabilizing Matrix"]
    item_ids = []
    legend = []
    ticker_histories = []
    normalize = False

    with open("H:/programming/gw2 tp data/datawars2/api-datawars2-ie_gw2_v1_items.csv",'r') as ticker_csv:
        lines = ticker_csv.readlines()
        for line in lines[1:]:
            splt = line.split(",")
            item_id = int(splt[0])
            item_name = splt[1]
            for i in range(len(desired_items)):
                if desired_items[i] == item_name:
                    item_ids.append(item_id)
                    print(desired_items[i], "->", item_id)
                    legend.append(item_name)
                    ticker_histories.append(np.load("H:/programming/gw2 tp data/datawars2/clean_v1_val_tickers/{}.npy".format(item_id)))
                    break

            if len(desired_items) == len(item_ids):
                break

    for cleaned_ticker_history in ticker_histories:
        sell_price_history = cleaned_ticker_history[:, ticker_functions.SELL_QUANTITY_MAX_IDX]

        coef = 1/10000
        if normalize:
            coef /= sell_price_history.max()

        graph.plot_data(sell_price_history*coef, clear=False)

    graph.set_legend(legend)
    if len(desired_items) == 1:
        plot_name = "{}_sell_prices".format(desired_items[0])
    else:
        plot_name = "item_sell_prices"

    if normalize:
        yLabel = "Normalized Min Sell Price"
    else:
        yLabel = "Min Sell Price (Gold)"

    graph.save_plot(plot_name, yLabel=yLabel, xLabel="Days Tracked")


def download_and_process_tickers():
    json_save_path = "H:/programming/gw2 tp data/datawars2/v1/unprocessed_json_data"
    numpy_save_path = "H:/programming/gw2 tp data/datawars2/v1/processed_json_data"
    training_data_save_path = "H:/programming/gw2 tp data/datawars2/v1/training_data"
    val_data_save_path = "H:/programming/gw2 tp data/datawars2/v1/validation_data"

    val_data_items = ["Mystic Coin",
                      "Glob of Ectoplasm",
                      "Bolt of Damask",
                      "Deldrimor Steel Ingot",
                      "Vicious Fang",
                      "Stabilizing Matrix",
                      "Superior Sigil of Force",
                      "Superior Rune of the Scholar",
                      "Cured Thick Leather Square",
                      "Platinum Ore"]

    download_more_detailed_tp_data(json_save_path)
    ticker_functions.parse_and_clean_all_tickers(folder_path=json_save_path, save_path=numpy_save_path)
    ticker_functions.create_training_data(parsed_data_path=numpy_save_path,
                                          training_data_save_path=training_data_save_path,
                                          val_data_save_path=val_data_save_path,
                                          val_data_item_names=val_data_items)

def main():
    # download_and_process_tickers()
    # train()
    validate()
    # ticker_functions.graph_train_data()


if __name__ == "__main__":
    main()