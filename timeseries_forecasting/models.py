from __future__ import annotations

import torch
from torch import nn


class RNNForecast(nn.Module):
    def __init__(
        self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int,
        pred_len: int,
    ) -> None:
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_size, output_size * pred_len)
        self.pred_len = pred_len
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(x)
        last_hidden = outputs[:, -1, :]
        projected = self.projection(last_hidden)
        return projected.view(-1, self.pred_len, self.output_size)


def build_model(
    model_name: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    output_size: int,
    pred_len: int,
) -> nn.Module:
    return RNNForecast(
        rnn_type=model_name,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=output_size,
        pred_len=pred_len,
    )
