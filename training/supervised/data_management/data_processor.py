import numpy as np
from MPFramework import MPFProcess, MPFProcessHandler
import time
from utils import ticker_functions, math_helpers


class DataProcessor(object):
    TICKER_LENGTH = 16

    def __init__(self, n_tickers, seq_len, trade_duration, batch_size, n_processes=12):
        self.processes = []
        self._init_processes(n_tickers, seq_len, trade_duration, batch_size, n_processes)
        self.ready_batches = []

    def get_random_batch(self):
        # print("get random batch")
        if len(self.ready_batches) == 0:
            for handler in self.processes:
                handler.put(header="batch_processed", data=None)

        while len(self.ready_batches) == 0:
            # print("checking for new batches")
            new_batches = []
            for handler in self.processes:
                data = handler.get_all(block=True, timeout=1)
                if data is not None:
                    new_batches += data

            # print("received:",len(new_batches))
            while len(new_batches) == 0:
                time.sleep(0.01)
                for handler in self.processes:
                    data = handler.get_all(block=True, timeout=1)
                    if data is not None:
                        new_batches += data
                    # else:
                    #     print("Data is None")

            # print("looping through",len(new_batches),"new batches")
            for data in new_batches:
                header, batch = data
                # print(np.shape(batch[0]), np.shape(batch[1]))
                self.ready_batches.append(batch)

        # print("returning a batch")
        return self.ready_batches.pop(0)


    def _init_processes(self, n_tickers, seq_len, trade_duration, batch_size, n_processes):
        for i in range(n_processes):
            handler = MPFProcessHandler()
            process = BatchingProcess("batching_process_{}".format(i), i)
            handler.setup_process(process)
            handler.put("initialization_data", (n_tickers, seq_len, trade_duration, batch_size))
            self.processes.append(handler)

    def cleanup(self):
        for handler in self.processes:
            handler.stop()
        del self.ready_batches



class BatchingProcess(MPFProcess):
    def __init__(self, name, delay_amount, loop_wait_time=None):
        super().__init__(name, loop_wait_time)
        self.data = None
        self.seq_len = None
        self.batch_size = None
        self.n_tickers = None
        self.tickers = None
        self.rng = None
        self.prepare_batch = False
        self.available_batches = []
        self.batch_num = 0
        self.n_batches_in_data = 0
        self.n_tickers_to_load = -1
        self.trade_duration = 0
        self.wait_period = 14
        self.mean = 0
        self.std = 1
        self.delay_amount = delay_amount

    def init(self):
        import numpy
        self.rng = numpy.random.RandomState(123)
        self.task_checker.wait_for_initialization(header="initialization_data")
        self.n_tickers, self.seq_len, self.trade_duration, self.batch_size = self.task_checker.latest_data

        print("Batching process configuring...\nn_tickers: {}\nseq_len: {}\nbatch_size: {}".format(self.n_tickers,
                                                                                                   self.seq_len,
                                                                                                   self.batch_size))

        self._refresh_data()
        self.prepare_batch = True

    def update(self, header, data):
        if header == "batch_processed":
            self.prepare_batch = True

    def step(self):
        # print("Preparing batch...")
        if not self.prepare_batch and len(self.available_batches) > 1:
            return

        batch_size = self.batch_size
        batch_x = []
        batch_y = []

        for i in range(batch_size):
            sample, label, quotient = None, None, None
            while sample is None:
                sample, label = self._get_random_sample()

            batch_x.append(sample)
            batch_y.append(label)

        # batch_x, batch_y, quotients = self._get_random_sample()

        self.available_batches.append((np.asarray(batch_x), np.asarray(batch_y)))
        self.batch_num += batch_size

        if self.batch_num > self.n_batches_in_data*10 and self.n_tickers_to_load != -1:
            self._refresh_data()

        # print("Batch ready")

    def _get_random_sample(self):
        sequence_inputs = []
        sequence_labels = []

        ticker_indices = [i for i in range(len(self.tickers))]
        self.rng.shuffle(ticker_indices)
        n_selected = 0

        for idx in ticker_indices:
            ticker = self.tickers[idx]
            history, labels = self.data[ticker]

            max_valid_start_day = len(history) - self.seq_len - self.trade_duration
            if max_valid_start_day < 0:
                continue

            indices = [i for i in range(max_valid_start_day)]
            start_day = self.rng.choice(indices)

            sequence = history[start_day:start_day + self.seq_len + self.trade_duration]
            label = labels[start_day:start_day + self.seq_len + self.trade_duration]
            sequence_inputs.append(sequence)
            sequence_labels.append(label)

            n_selected += 1
            if n_selected == self.n_tickers:
                break

        if n_selected < self.n_tickers:
            return None, None, None

        # print(np.shape(sequence_inputs))
        # print(np.concatenate(sequence_inputs, axis=-1).shape)
        # print()
        labels = np.concatenate(sequence_labels, axis=-1)
        labels = np.where(labels == -1, -3, labels)
        labels = np.where(labels > 3, 3, labels)
        return np.concatenate(sequence_inputs, axis=-1), labels


    def _refresh_data(self):
        print("COLLECTED {} OF {} BATCHES, LOADING NEW TICKER SET!".format(self.batch_num, self.n_batches_in_data))
        from training.supervised.data_management import data_loader
        if self.data is not None:
            self.data.clear()

        self.batch_num = 0
        self.data, shortest_length = data_loader.load_random_tickers("H:/programming/gw2 tp data/datawars2/v1_training_data", self.n_tickers_to_load)
        self.n_batches_in_data = self.n_tickers_to_load*shortest_length // self.batch_size
        self.tickers = list(self.data.keys())

        # self.mean, self.std = ticker_functions.get_known_ticker_stats()
        # self._compute_start_day_probability_list()
        print("LOADED {} BATCHES".format(self.n_batches_in_data))

    def _compute_start_day_probability_list(self):
        seq_len = self.seq_len + self.trade_duration

        for ticker, history in self.data.items():
            logits = []

            for i in range(0, len(history) - seq_len - self.wait_period):
                start = i + seq_len
                stop = start + self.wait_period
                trade_window = history[start:stop]
                n_positive = 0
                n_negative = 0
                for j in range(len(trade_window)-1):
                    slope = trade_window[j+1] - trade_window[j]
                    if slope > 0:
                        n_positive += 1
                    elif slope < 0:
                        n_negative += 1
                score = n_negative - n_positive
                logits.append(score)
            probs = math_helpers.softmax(logits)
            self.data[ticker] = (history, probs)


    def publish(self):
        if self.prepare_batch:
            # print("PUBLISHING",numpy.shape(batch[0]), numpy.shape(batch[1]))
            self.results_publisher.publish(header="batch", data=self.available_batches.pop(0))
            self.prepare_batch = False

    def cleanup(self):
        if self.data is not None:
            self.data.clear()

        del self.data
        del self.available_batches
