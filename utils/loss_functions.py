import torch
import numpy as np
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def quotient_prediction_loss(model, batch_inputs, batch_quotients, seq_len, trade_duration):
    inputs = torch.as_tensor(batch_inputs, dtype=torch.float64).to(device)
    quotients = torch.as_tensor(batch_quotients, dtype=torch.float64).to(device)

    model_out = model(inputs)
    loss = torch.nn.MSELoss()(model_out, quotients)

    indices0 = model_out[0][-1].argsort().__reversed__()
    indices1 = model_out[-1][-1].argsort().__reversed__()
    print([round(arg, 3) for arg in model_out[0][-1][indices0].tolist()], " | ",
          [round(arg, 3) for arg in model_out[-1][-1][indices1].tolist()])
    print([round(arg, 3) for arg in quotients[0][-1][indices0].tolist()], " | ",
          [round(arg, 3) for arg in quotients[-1][-1][indices1].tolist()])
    print()

    return loss.mean()



def geo_mean_loss(model, batch_inputs, batch_quotients, seq_len, trade_duration):
    batch_inputs = torch.as_tensor(batch_inputs, dtype=torch.float64).to(device)
    batch_quotients = torch.as_tensor(batch_quotients, dtype=torch.float64).to(device)

    model_out = model(batch_inputs)
    penalties = 1 + (model_out[:, seq_len:, :] * batch_quotients[:, seq_len:, :]).sum(dim=-1, dtype=torch.float64)
    clipped = torch.where(penalties < 0, penalties + penalties.detach().abs() + 1e-12, penalties)
    products = torch.exp(torch.log(clipped).mean(dim=-1))
    # products = torch.prod(clipped, dim=-1)
    loss = products
    # loss = products.pow(1.0 / trade_duration)

    indices0 = model_out[0][-1].argsort().__reversed__()
    indices1 = model_out[1][-1].argsort().__reversed__()
    print([round(arg, 3) for arg in model_out[0][-1][indices0].tolist()], " | ",
          [round(arg, 3) for arg in model_out[1][-1][indices1].tolist()])
    print([round(arg, 3) for arg in batch_quotients[0][-1][indices0].tolist()], " | ",
          [round(arg, 3) for arg in batch_quotients[1][-1][indices1].tolist()])
    print()

    return -loss.mean()

def tvd_loss(model, batch_inputs, batch_quotients):
    quotients = torch.as_tensor(batch_quotients, dtype=torch.float64).to(device)

    batch_size, seq_len, _ = quotients.shape
    sft = torch.nn.Softmax(dim=-1)

    probs = sft(quotients)
    clipped = torch.where(quotients < 0, torch.zeros_like(quotients), probs)
    label = clipped / (1e-12 + clipped.sum(dim=-1).view(batch_size, seq_len, 1))

    model_out = model(torch.as_tensor(batch_inputs, dtype=torch.float64).to(device)).view_as(quotients)


    # loss = (model_out - label).abs().sum(dim=-1)
    loss = torch.nn.KLDivLoss()(model_out, label)
    indices0 = model_out[0][-1].argsort().__reversed__()
    indices1 = model_out[1][-1].argsort().__reversed__()
    print([round(arg, 3) for arg in model_out[0][-1][indices0].tolist()], " | ", [round(arg, 3) for arg in model_out[1][-1][indices1].tolist()])
    print([round(arg, 3) for arg in label[0][-1][indices0].tolist()], " | ", [round(arg, 3) for arg in label[1][-1][indices1].tolist()])
    print()


    return loss.mean()
