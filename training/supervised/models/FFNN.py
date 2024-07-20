import torch
import torch.nn as nn
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class FFNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.dims = [input_shape,2048,1024,512,256,128,64,32,16,output_shape]

        self.input_shape = input_shape
        self.output_shape = output_shape

        layers = []
        for i in range(len(self.dims)-1):
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))

            if i != len(self.dims)-2:
                layers.append(nn.ReLU())
        layers.append(nn.Softmax(dim=-1))

        self.model = nn.Sequential(*layers).to(device)


    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.as_tensor(x, dtype=torch.float32).to(device)

        return self.model(x)