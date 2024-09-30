import MuyGPyS._src.math.torch as torch


def _rbf_fn(squared_dists: torch.ndarray, **kwargs) -> torch.ndarray:
    return torch.exp(-squared_dists / 2.0)


def _matern_05_fn(dists: torch.ndarray, **kwargs) -> torch.ndarray:
    return torch.exp(-dists)


def _matern_15_fn(dists: torch.ndarray, **kwargs) -> torch.ndarray:
    K = dists * torch.sqrt(torch.array(3))
    return (1.0 + K) * torch.exp(-K)


def _matern_25_fn(dists: torch.ndarray, **kwargs) -> torch.ndarray:
    K = dists * torch.sqrt(torch.array(5))
    return (1.0 + K + K**2 / 3.0) * torch.exp(-K)


def _matern_inf_fn(dists: torch.ndarray, **kwargs) -> torch.ndarray:
    return torch.exp(-(dists**2) / 2.0)


def _matern_gen_fn(
    dists: torch.ndarray, smoothness: float, **kwargs
) -> torch.ndarray:
    raise NotImplementedError(
        f'Function "matern_gen_fn" does not support smoothness={smoothness}. '
        f"Torch only supports 1/2, 3/2, 5/2 and torch.inf."
    )
