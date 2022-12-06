import torch


def _cross_entropy_fn(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ll_eps: float = 1e-15,
) -> float:
    one_hot_targets = torch.where(targets > 0.0, 1.0, 0.0)
    softmax_predictions = torch.nn.Softmax(predictions, axis=1)

    return torch.nn.CrossEntropyLoss(
        one_hot_targets, softmax_predictions, eps=ll_eps, normalize=False
    )


def _mse_fn_unnormalized(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    return torch.sum((predictions - targets) ** 2)


def _mse_fn(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    batch_count, response_count = predictions.shape
    return _mse_fn_unnormalized(predictions, targets) / (
        batch_count * response_count
    )


def _lool_fn(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    variances: torch.Tensor,
    sigma_sq: torch.Tensor,
) -> float:
    return torch.sum(
        (predictions - targets) ** 2 / (sigma_sq * variances)
    ) + torch.sum(torch.log(sigma_sq * variances))
