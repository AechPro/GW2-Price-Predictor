import torch
import torch.nn as nn
from utils.math_helpers import SigmoidDistribution
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class LSTM(nn.Module):
    def __init__(self, input_shape, output_features):
        super().__init__()
        self.output_fn = None

        ##
        # ff1 = 3000
        # ff2 = 2500
        # ff3 = 2000
        # ff4 = 1500
        # ff5 = 1000
        # self.lstm1 = 3000
        # ff6 = 500


        ff1 = 5000
        ff2 = 4000
        ff3 = 3000
        ff4 = 2000
        ff5 = 1500
        self.lstm1 = 4000
        ff6 = 1000
        input_features, seq_len = input_shape

        self.layer1 = nn.Linear(in_features=input_features, out_features=ff1).double().to(device)
        self.layer2 = nn.Linear(in_features=ff1, out_features=ff2).double().to(device)
        self.layer3 = nn.Linear(in_features=ff2, out_features=ff3).double().to(device)
        self.layer4 = nn.Linear(in_features=ff3, out_features=ff4).double().to(device)
        self.layer5 = nn.Linear(in_features=ff4+input_features, out_features=ff5).double().to(device)
        self.lstm = nn.LSTM(ff5, self.lstm1, 1, batch_first=True).double().to(device)
        self.layer6 = nn.Linear(in_features=self.lstm1, out_features=ff6).double().to(device)
        self.output_layer = nn.Linear(in_features=ff6, out_features=output_features).double().to(device)

        self.act_fn = nn.ReLU()
        # self.output_fn = nn.Softmax(dim=-1)
        # self.output_fn = SigmoidDistribution()
        # self.output_fn = nn.Sigmoid()
        print("LOADED MODEL WITH {} TRAINABLE PARAMETERS".format(torch.nn.utils.parameters_to_vector(self.parameters()).numel()))

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.as_tensor(x, dtype=torch.float64).to(device)

        h0 = torch.zeros(1, x.size(0), self.lstm1, dtype=torch.float64).to(device)
        c0 = torch.zeros(1, x.size(0), self.lstm1, dtype=torch.float64).to(device)

        l1 = self.act_fn(self.layer1(x))
        l2 = self.act_fn(self.layer2(l1))
        l3 = self.act_fn(self.layer3(l2))
        l4 = self.act_fn(self.layer4(l3))
        l5 = self.act_fn(self.layer5(torch.cat((l4, x), dim=-1)))

        lstm_out, (hn, cn) = self.lstm(l5, (h0, c0))
        l6 = self.act_fn(self.layer6(lstm_out))
        output = self.output_layer(l6)
        if self.output_fn is not None:
            output = self.output_fn(output)

        return output