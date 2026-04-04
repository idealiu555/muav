import numpy as np
import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    """Running mean/std normalization for critic targets."""

    def __init__(
        self,
        input_shape: int,
        *,
        beta: float = 0.99999,
        epsilon: float = 1e-5,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.beta = beta
        self.epsilon = epsilon
        self.device_spec = torch.device(device)

        self.running_mean = nn.Parameter(torch.zeros(input_shape, dtype=torch.float32), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape, dtype=torch.float32), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)

        self.to(self.device_spec)

    def running_mean_var(self) -> tuple[torch.Tensor, torch.Tensor]:
        debias = self.debiasing_term.clamp(min=self.epsilon)
        mean = self.running_mean / debias
        mean_sq = self.running_mean_sq / debias
        var = (mean_sq - mean.pow(2)).clamp(min=1e-2)
        return mean, var

    def _as_tensor(self, value: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        return value.to(self.device_spec, dtype=torch.float32)

    @torch.no_grad()
    def update(self, value: torch.Tensor | np.ndarray) -> None:
        tensor = self._as_tensor(value)
        batch_mean = tensor.mean(dim=0)
        batch_sq_mean = tensor.pow(2).mean(dim=0)

        self.running_mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    def normalize(self, value: torch.Tensor | np.ndarray) -> torch.Tensor:
        tensor = self._as_tensor(value)
        mean, var = self.running_mean_var()
        return (tensor - mean) / torch.sqrt(var)

    def denormalize(self, value: torch.Tensor | np.ndarray) -> torch.Tensor:
        tensor = self._as_tensor(value)
        mean, var = self.running_mean_var()
        return tensor * torch.sqrt(var) + mean
