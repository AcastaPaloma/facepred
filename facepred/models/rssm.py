"""Recurrent state-space model used by FacePred."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RSSMOutput:
    """Container for RSSM sequence outputs."""

    deterministic: torch.Tensor
    stochastic: torch.Tensor
    prior_logits: torch.Tensor | None = None
    posterior_logits: torch.Tensor | None = None
    prior_mean: torch.Tensor | None = None
    prior_log_std: torch.Tensor | None = None
    posterior_mean: torch.Tensor | None = None
    posterior_log_std: torch.Tensor | None = None

    @property
    def state(self) -> torch.Tensor:
        return torch.cat([self.deterministic, self.stochastic], dim=-1)


class RSSM(nn.Module):
    """Small Dreamer-style RSSM with categorical or Gaussian latents."""

    def __init__(
        self,
        input_dim: int,
        gru_hidden: int = 128,
        latent_type: str = "categorical",
        categorical_classes: int = 16,
        categorical_dims: int = 8,
        gaussian_dim: int = 32,
        use_stochastic: bool = True,
    ) -> None:
        super().__init__()
        if latent_type not in {"categorical", "gaussian"}:
            raise ValueError(f"Unsupported latent_type: {latent_type}")

        self.input_dim = input_dim
        self.gru_hidden = gru_hidden
        self.latent_type = latent_type
        self.categorical_classes = categorical_classes
        self.categorical_dims = categorical_dims
        self.gaussian_dim = gaussian_dim
        self.use_stochastic = use_stochastic
        self.latent_dim = (
            categorical_classes * categorical_dims if latent_type == "categorical" else gaussian_dim
        )
        if not use_stochastic:
            self.latent_dim = 0

        self.gru = nn.GRUCell(input_dim + self.latent_dim, gru_hidden)

        if use_stochastic and latent_type == "categorical":
            out_dim = categorical_classes * categorical_dims
            self.prior = nn.Linear(gru_hidden, out_dim)
            self.posterior = nn.Linear(gru_hidden + input_dim, out_dim)
        elif use_stochastic:
            self.prior = nn.Linear(gru_hidden, gaussian_dim * 2)
            self.posterior = nn.Linear(gru_hidden + input_dim, gaussian_dim * 2)
        else:
            self.prior = nn.Identity()
            self.posterior = nn.Identity()

    @property
    def state_dim(self) -> int:
        return self.gru_hidden + self.latent_dim

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.gru_hidden, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    def forward(
        self,
        inputs: torch.Tensor,
        initial: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> RSSMOutput:
        """Run the RSSM over ``inputs`` shaped ``[batch, steps, input_dim]``."""
        if inputs.ndim != 3:
            raise ValueError(f"inputs must be [batch, steps, dim], got {tuple(inputs.shape)}")
        batch, steps, _ = inputs.shape
        h, z_prev = initial if initial is not None else self.initial_state(batch, inputs.device)

        deterministic: list[torch.Tensor] = []
        stochastic: list[torch.Tensor] = []
        prior_logits: list[torch.Tensor] = []
        posterior_logits: list[torch.Tensor] = []
        prior_mean: list[torch.Tensor] = []
        prior_log_std: list[torch.Tensor] = []
        posterior_mean: list[torch.Tensor] = []
        posterior_log_std: list[torch.Tensor] = []

        for t in range(steps):
            x_t = inputs[:, t]
            gru_input = torch.cat([x_t, z_prev], dim=-1) if self.use_stochastic else x_t
            h = self.gru(gru_input, h)
            deterministic.append(h)

            if not self.use_stochastic:
                z_t = torch.empty(batch, 0, device=inputs.device)
            elif self.latent_type == "categorical":
                p_logits = self.prior(h).view(batch, self.categorical_dims, self.categorical_classes)
                q_logits = self.posterior(torch.cat([h, x_t], dim=-1)).view(
                    batch, self.categorical_dims, self.categorical_classes
                )
                z_t = self._sample_categorical(q_logits)
                prior_logits.append(p_logits)
                posterior_logits.append(q_logits)
            else:
                p_mean, p_log_std = self._split_gaussian(self.prior(h))
                q_mean, q_log_std = self._split_gaussian(self.posterior(torch.cat([h, x_t], dim=-1)))
                z_t = self._sample_gaussian(q_mean, q_log_std)
                prior_mean.append(p_mean)
                prior_log_std.append(p_log_std)
                posterior_mean.append(q_mean)
                posterior_log_std.append(q_log_std)

            stochastic.append(z_t)
            z_prev = z_t.detach()

        deterministic_t = torch.stack(deterministic, dim=1)
        stochastic_t = torch.stack(stochastic, dim=1)

        if self.latent_type == "categorical" and self.use_stochastic:
            return RSSMOutput(
                deterministic=deterministic_t,
                stochastic=stochastic_t,
                prior_logits=torch.stack(prior_logits, dim=1),
                posterior_logits=torch.stack(posterior_logits, dim=1),
            )
        if self.use_stochastic:
            return RSSMOutput(
                deterministic=deterministic_t,
                stochastic=stochastic_t,
                prior_mean=torch.stack(prior_mean, dim=1),
                prior_log_std=torch.stack(prior_log_std, dim=1),
                posterior_mean=torch.stack(posterior_mean, dim=1),
                posterior_log_std=torch.stack(posterior_log_std, dim=1),
            )
        return RSSMOutput(deterministic=deterministic_t, stochastic=stochastic_t)

    def imagine(
        self,
        horizon_steps: int,
        initial: tuple[torch.Tensor, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> RSSMOutput:
        """Roll the prior forward with zero inputs or a provided input/action."""
        h, z_prev = initial
        batch = h.shape[0]
        inputs = torch.zeros(batch, horizon_steps, self.input_dim, device=h.device)
        if action is not None:
            if action.ndim == 2:
                action = action[:, None, :].expand(-1, horizon_steps, -1)
            inputs[..., : action.shape[-1]] = action
        return self.forward(inputs, initial=(h, z_prev))

    def _sample_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            sample = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        else:
            indices = logits.argmax(dim=-1)
            sample = F.one_hot(indices, num_classes=self.categorical_classes).float()
        return sample.flatten(start_dim=-2)

    def _split_gaussian(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = params.chunk(2, dim=-1)
        log_std = raw_std.clamp(-5.0, 2.0)
        return mean, log_std

    def _sample_gaussian(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mean
        return mean + torch.randn_like(mean) * log_std.exp()
