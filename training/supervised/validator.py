import numpy as np
import torch
from training.supervised.data_management import data_loader, DataProcessor
from training.supervised.models import LSTM
from utils import graphing_helper as graph, ticker_functions, math_helpers
import itertools
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Validator(object):
    def __init__(self, seq_len, ticker_length, n_tickers, model=None):
        self.seq_len = seq_len
        self.ticker_length = ticker_length
        self.n_tickers = n_tickers
        self.inputs = None
        self.quotients = None
        if model is None:
            self.model = LSTM((self.ticker_length*self.n_tickers, self.seq_len), self.n_tickers)
            self.model.load_state_dict(torch.load("E:/programming/gw2_bot_data/models/quotient_134M_48100/MODEL.pt"))
        else:
            self.model = model

    def permute_inputs(self, inputs, permutation, permuted=None):
        needs_return = False
        if permuted is None:
            permuted = torch.zeros_like(inputs)
            needs_return = True

        ticker_length = self.ticker_length
        for i in range(len(permutation)):
            base_start = i*ticker_length
            base_stop = base_start + ticker_length

            permuted_start = permutation[i]*ticker_length
            permuted_stop = permuted_start + ticker_length
            permuted[:, :, permuted_start:permuted_stop] = inputs[:, :, base_start:base_stop]

        if needs_return:
            return permuted

    def test_model(self, n_training_data_tests=1):
        scores = []
        for i in range(n_training_data_tests):
            self.prepare_data()
            scores.append(self._test())

        print("TEST SCORES",scores)
        print("AVERAGE SCORE",np.mean(scores))

    @torch.no_grad()
    def _test(self):
        self.model.eval()
        max_permutations_per_compute = 100
        seq_len = self.seq_len
        inputs = torch.as_tensor(self.inputs, dtype=torch.float64).view(1, -1, 160).to(device)
        quotients = torch.as_tensor(self.quotients, dtype=torch.float64).view(1, -1, 10).to(device)
        model = self.model
        initial_gold = 1000
        gold_per_trade = 100
        n_permutations = 150

        indices = [i for i in range(self.n_tickers)]
        permutations = list(itertools.permutations(indices))
        np.random.shuffle(permutations)

        desired_shape = (quotients.shape[1] - seq_len, self.n_tickers)
        avg_model_prediction = torch.zeros(desired_shape).to(device)

        batched_permutations = []
        n_permutations_computed = 0
        for perm in range(n_permutations):
            permutation = permutations[perm]
            permuted_inputs = self.permute_inputs(inputs, permutation)
            batched_permutations.append(permuted_inputs)
            if len(batched_permutations) >= max_permutations_per_compute:
                print("computing inference...")
                batched_permutations = torch.stack(batched_permutations).view(len(batched_permutations),
                                                                              -1,
                                                                              self.ticker_length * self.n_tickers)
                model_out = model(batched_permutations.to(device))[:, seq_len:, :]
                avg_model_prediction += model_out.sum(dim=0).view(-1, self.n_tickers)
                n_permutations_computed += len(batched_permutations)
                print("computed",n_permutations_computed,"permutations so far")
                batched_permutations = []

        if len(batched_permutations) > 0:
            print("computing final", len(batched_permutations), "permutations")
            batched_permutations = torch.stack(batched_permutations).view(len(batched_permutations), -1, self.ticker_length*self.n_tickers)
            model_out = model(batched_permutations)[:, seq_len:, :]
            avg_model_prediction += model_out.sum(dim=0).view(-1, self.n_tickers)
            n_permutations_computed += len(batched_permutations)

        avg_model_prediction /= n_permutations_computed
        print("average model prediction computed after",n_permutations_computed,"input permutations")

        zeroed = torch.where(avg_model_prediction < 0, torch.zeros_like(avg_model_prediction), avg_model_prediction)
        s = torch.sum(zeroed, dim=-1)
        coef = torch.where(s > 1, 1/s, torch.ones_like(s))
        normalized = zeroed * coef.unsqueeze(1).expand(-1, 10)
        penalties = (normalized * quotients[:, seq_len:, :]).sum(dim=-1, dtype=torch.float64)

        # for i in range(len(penalties[0])):
        #     print(math_helpers.round_list(avg_model_prediction[i].cpu().tolist()))
        #     print(math_helpers.round_list(quotients[0, seq_len:, :][i].cpu().tolist()))
        #     print(penalties[0][i].item())
        #     print()

        gold_earned_per_trade = gold_per_trade*penalties

        total_profits = gold_earned_per_trade.sum(dim=-1).item()
        print("Initial gold:",initial_gold)
        print("Days traded:", gold_earned_per_trade.shape[-1])
        print("Gold earned:", total_profits)
        print("Final gold:",total_profits+initial_gold)
        print("Trade results:",math_helpers.round_list(gold_earned_per_trade.flatten().cpu().tolist()))
        print()

        self.model.train()
        return total_profits

    def prepare_data(self):
        slice_at = self.seq_len + 30
        # data, shortest = data_loader.load_val_data("H:\\programming\\gw2 tp data\\datawars2\\v1\\validation_data", slice_at=slice_at)
        data, shortest = data_loader.load_random_tickers("H:\\programming\\gw2 tp data\\datawars2\\v1\\training_data", n_tickers=10, slice_at=slice_at)

        inputs = []
        quotients = []
        for i in range(shortest):
            concatenated_tickers = []
            concatenated_quotients = []

            for ticker in data.keys():
                ticker_data, ticker_trade_quotients = data[ticker]
                current_day = ticker_data[i]
                concatenated_tickers += current_day.tolist()
                concatenated_quotients.append(ticker_trade_quotients[i])

            inputs.append(concatenated_tickers)
            quotients.append(concatenated_quotients)

        self.inputs = np.asarray(inputs)
        self.quotients = np.asarray(quotients).reshape(-1, self.n_tickers)

