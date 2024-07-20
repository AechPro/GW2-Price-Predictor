import os
import numpy as np
from utils import WelfordRunningStat, ticker_functions


def _load_raw_tickers_from(folder_path, desired_tickers=None, smooth=True):
    shortest_length = np.inf
    num_samples = 0
    all_data = {}
    stats = WelfordRunningStat(16)
    for fname in os.listdir(folder_path):
        if ".npy" not in fname:
            continue

        if desired_tickers is not None:
            if fname not in desired_tickers:
                continue

        data = np.load(os.path.join(folder_path, fname))
        n = len(data)
        stats.increment(data, n)
        num_samples += n
        if n < shortest_length:
            shortest_length = n
        item_id = int(fname[:fname.find(".npy")])
        all_data[item_id] = data

    if smooth:
        all_data = ticker_functions.smooth_tickers(all_data)

    return all_data, shortest_length, stats

def _load_training_data_from(folder_path, desired_tickers=None, slice_at=None):
    all_data = {}
    shortest = np.inf
    x_path = os.path.join(folder_path, "inputs")
    y_path = os.path.join(folder_path, "labels")

    for fname in os.listdir(x_path):
        if ".npy" not in fname:
            continue

        if desired_tickers is not None:
            if fname not in desired_tickers:
                continue

        x = np.load(os.path.join(x_path, fname))
        y = np.load(os.path.join(y_path, fname))

        if slice_at is not None and len(x) > slice_at:
            x = x[-slice_at:]
            y = y[-slice_at:]

        if len(x) < shortest:
            shortest = len(x)

        item_id = int(fname[:fname.find(".npy")])
        all_data[item_id] = (x, y)

    return all_data, shortest

def load_val_data(folder_path, smooth=False, raw_data=False, slice_at=None):
    if raw_data:
        return _load_raw_tickers_from(folder_path, smooth=smooth)

    all_data, shortest = _load_training_data_from(folder_path, slice_at=slice_at)

    print("Loaded {} validation samples".format(len(all_data.keys())))
    return all_data, shortest

def load_random_tickers(folder_path, n_tickers, smooth=True, raw_data=False, slice_at=None):
    x_path = os.path.join(folder_path, "inputs")
    all_tickers = []
    for fname in os.listdir(x_path):
        all_tickers.append(fname)

    if n_tickers == -1:
        tickers = all_tickers
    else:
        tickers = np.random.choice(all_tickers, n_tickers, replace=False)

    if raw_data:
        return _load_raw_tickers_from(folder_path, smooth=smooth, desired_tickers=tickers)

    all_data, shortest = _load_training_data_from(folder_path, desired_tickers=tickers, slice_at=slice_at)
    print("Loaded {} training samples".format(len(all_data.keys())))
    return all_data, shortest


def load_specific_tickers(folder_path, tickers, smooth=True, raw_data=False, slice_at=None):
    if raw_data:
        return _load_raw_tickers_from(folder_path, smooth=smooth, desired_tickers=tickers)

    all_data, shortest = _load_training_data_from(folder_path, desired_tickers=tickers, slice_at=slice_at)
    print("Loaded {} training samples".format(len(all_data.keys())))
    return all_data, shortest

