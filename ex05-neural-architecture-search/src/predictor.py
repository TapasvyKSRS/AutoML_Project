from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


from collections import OrderedDict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def accuracy_mse(prediction, target, scale=100.0):
    prediction = prediction.detach() * scale
    target = (target) * scale
    return F.mse_loss(prediction, target)


class NamedAverageMeter:
    """Computes and stores the average and current value, ported from naszilla repo"""

    def __init__(self, name, fmt=":f"):
        """
        Initialization of AverageMeter
        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = "{name}: {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class AverageMeterGroup:
    """Average meter group for multiple average meters, ported from Naszilla repo."""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = NamedAverageMeter(k, ":4f")
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        return "  ".join(v.summary() for v in self.meters.values())


class FeedforwardNet(nn.Module):
    def __init__(
        self,
        input_dims: int = 5,
        num_layers: int = 3,
        layer_width: list = [10, 10, 10],
        output_dims: int = 1,
        activation="relu",
    ):
        super(FeedforwardNet, self).__init__()
        assert (
            len(layer_width) == num_layers
        ), "number of widths should be \
        equal to the number of layers"

        self.activation = eval("F." + activation)

        all_units = [input_dims] + layer_width
        self.layers = nn.ModuleList(
            [
                nn.Linear(all_units[i], all_units[i + 1]) for i in range(num_layers)
            ]
        )

        self.out = nn.Linear(all_units[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.out(x)


class MLPPredictor:
    def __init__(self):
        self.hyperparams = {
            "num_layers": 5,
            "layer_width": 20,
            "batch_size": 32,
            "lr": 0.001,
            "regularization": 0.2,
        }

    def get_model(self, **kwargs):
        predictor = FeedforwardNet(**kwargs)
        return predictor

    def fit(self, xtrain, ytrain, train_info=None, epochs=50, loss="mae", verbose=0):
        num_layers = self.hyperparams["num_layers"]
        layer_width = self.hyperparams["layer_width"]
        batch_size = self.hyperparams["batch_size"]
        lr = self.hyperparams["lr"]
        regularization = self.hyperparams["regularization"]

        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)

        X_tensor = torch.FloatTensor(xtrain).to(device)
        y_tensor = torch.FloatTensor(ytrain).to(device)
        train_data = TensorDataset(X_tensor, y_tensor)

        data_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=False,
        )

        self.model = self.get_model(
            input_dims=xtrain.shape[1],
            num_layers=num_layers,
            layer_width=num_layers * [layer_width],
        )
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.99))

        if loss == "mse":
            criterion = nn.MSELoss().to(device)
        elif loss == "mae":
            criterion = nn.L1Loss().to(device)

        self.model.train()

        for e in range(epochs):
            meters = AverageMeterGroup()
            for b, batch in enumerate(data_loader):
                optimizer.zero_grad()
                input = batch[0].to(device)
                target = batch[1].to(device)
                prediction = self.model(input).view(-1)

                loss_fn = criterion(prediction, target)
                # add L1 regularization
                params = torch.cat(
                    [
                        x[1].view(-1)
                        for x in self.model.named_parameters()
                        if x[0] == "out.weight"
                    ]
                )
                loss_fn += regularization * torch.norm(params, 1)
                loss_fn.backward()
                optimizer.step()

                mse = accuracy_mse(prediction, target)
                meters.update(
                    {"loss": loss_fn.item(), "mse": mse.item()}, n=target.size(0)
                )

            if verbose and e % 100 == 0:
                print("Epoch {}, {}, {}".format(e, meters["loss"], meters["mse"]))

        return self

    def query(self, xtest, info=None, eval_batch_size=None):
        X_tensor = torch.FloatTensor(np.array(xtest)).to(device)
        test_data = TensorDataset(X_tensor)

        eval_batch_size = len(xtest) if eval_batch_size is None else eval_batch_size
        test_data_loader = DataLoader(
            test_data, batch_size=eval_batch_size, pin_memory=False
        )

        self.model.eval()
        pred = []
        with torch.no_grad():
            for _, batch in enumerate(test_data_loader):
                prediction = self.model(batch[0].to(device)).view(-1)
                pred.append(prediction.cpu().numpy())

        pred = np.concatenate(pred)
        return np.squeeze(pred)

    def set_random_hyperparams(self):

        params = {
            "num_layers": int(np.random.choice(range(5, 25))),
            "layer_width": int(np.random.choice(range(5, 25))),
            "batch_size": 32,
            "lr": np.random.choice([0.1, 0.01, 0.005, 0.001, 0.0001]),
            "regularization": 0.2,
        }

        self.hyperparams = params
        return params


class EnsemblePredictor:
    def __init__(self, n_models=3):
        self.predictors = [MLPPredictor() for _ in range(n_models)]
        for p in self.predictors:
            p.set_random_hyperparams()

    def fit(self, xtrain, ytrain, train_info=None, epochs=50, loss="mae", verbose=0):
        for p in self.predictors:
            p.fit(xtrain, ytrain, train_info, epochs, loss, verbose)

        return self

    def query(self, xtest, info=None, eval_batch_size=None):
        preds = []
        for p in self.predictors:
            preds.append(p.query(xtest, info, eval_batch_size))
        return np.mean(preds, axis=0)
