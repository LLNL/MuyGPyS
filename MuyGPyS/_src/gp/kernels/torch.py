import torch


def _rbf_fn(squared_dists: torch.Tensor, length_scale: float) -> torch.Tensor:
    return torch.exp(-squared_dists / (2 * length_scale**2))


def _matern_05_fn(dists: torch.Tensor, length_scale: float) -> torch.Tensor:
    dists = dists / length_scale
    return torch.exp(-dists)


def _matern_15_fn(dists: torch.Tensor, length_scale: float) -> torch.Tensor:
    dists = dists / length_scale
    K = dists * torch.sqrt(torch.tensor(3))
    return (1.0 + K) * torch.exp(-K)


def _matern_25_fn(dists: torch.Tensor, length_scale: float) -> torch.Tensor:
    dists = dists / length_scale
    K = dists * torch.sqrt(torch.tensor(5))
    return (1.0 + K + K**2 / 3.0) * torch.exp(-K)


def _matern_inf_fn(dists: torch.Tensor, length_scale: float) -> torch.Tensor:
    dists = dists / length_scale
    return torch.exp(-(dists**2) / 2.0)


def _matern_gen_fn(
    dists: torch.Tensor, nu: float, length_scale: float
) -> torch.Tensor:
    raise NotImplementedError(
        f'Function "matern_gen_fn" does not support values of nu other than 1/2, 3/2, 5/2 and torch.inf!'
    )
