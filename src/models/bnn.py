from __future__ import annotations

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer import Predictive
from pyro.nn import PyroModule, PyroSample

from src.config import (
    BNN_DROPOUT_RATE,
    BNN_HIDDEN_DIM_1,
    BNN_HIDDEN_DIM_2,
    BNN_PRIOR_SCALE,
    DEFAULT_BNN_MC_SAMPLES,
    DEVICE,
)

class BayesianMLP(PyroModule):
    """
    Bayesian MLP for binary classification using Pyro.

    Architecture:
        input -> hidden_1 -> activation -> dropout
              -> hidden_2 -> activation -> dropout
              -> output(logit)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = BNN_HIDDEN_DIM_1,
        hidden_dim_2: int = BNN_HIDDEN_DIM_2,
        prior_scale: float = BNN_PRIOR_SCALE,
        dropout_rate: float = BNN_DROPOUT_RATE,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.prior_scale = prior_scale
        self.dropout_rate = dropout_rate

        self.fc1 = PyroModule[nn.Linear](input_dim, hidden_dim_1)
        self.fc1.weight = PyroSample(
            dist.Normal(0.0, prior_scale)
            .expand([hidden_dim_1, input_dim])
            .to_event(2)
        )
        self.fc1.bias = PyroSample(
            dist.Normal(0.0, prior_scale)
            .expand([hidden_dim_1])
            .to_event(1)
        )

        self.fc2 = PyroModule[nn.Linear](hidden_dim_1, hidden_dim_2)
        self.fc2.weight = PyroSample(
            dist.Normal(0.0, prior_scale)
            .expand([hidden_dim_2, hidden_dim_1])
            .to_event(2)
        )
        self.fc2.bias = PyroSample(
            dist.Normal(0.0, prior_scale)
            .expand([hidden_dim_2])
            .to_event(1)
        )

        self.out = PyroModule[nn.Linear](hidden_dim_2, 1)
        self.out.weight = PyroSample(
            dist.Normal(0.0, prior_scale)
            .expand([1, hidden_dim_2])
            .to_event(2)
        )
        self.out.bias = PyroSample(
            dist.Normal(0.0, prior_scale)
            .expand([1])
            .to_event(1)
        )

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Pyro model forward.

        Args:
            x: shape [batch_size, input_dim]
            y: shape [batch_size]
        """
        hidden = self.activation(self.fc1(x))
        hidden = self.dropout(hidden)

        hidden = self.activation(self.fc2(hidden))
        hidden = self.dropout(hidden)

        logits = self.out(hidden).squeeze(-1)

        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

        return logits


@torch.no_grad()
def predict_proba_mc(
    model: BayesianMLP,
    guide,
    x: torch.Tensor,
    num_samples: int = DEFAULT_BNN_MC_SAMPLES,
    batch_size: int = 4096,
    device: str = DEVICE,
) -> np.ndarray:
    """
    Monte Carlo predictive probabilities for binary classification.

    Returns:
        Mean predictive probabilities of shape [n_examples].
    """
    model.eval()
    x = x.to(device)

    all_probs = []

    for start_idx in range(0, x.shape[0], batch_size):
        batch_x = x[start_idx : start_idx + batch_size]

        predictive = Predictive(
            model=model,
            guide=guide,
            num_samples=num_samples,
            return_sites=["_RETURN"],
        )
        samples = predictive(batch_x)
        logits_samples = samples["_RETURN"]  # [num_samples, batch]
        prob_samples = torch.sigmoid(logits_samples)
        mean_probs = prob_samples.mean(dim=0)

        all_probs.append(mean_probs.cpu())

    return torch.cat(all_probs, dim=0).numpy()
