import torch


def _cross_entropy_fn(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    loss = torch.nn.CrossEntropyLoss()
    one_hot_targets = torch.where(targets > 0.0, 1.0, 0.0)
    softmax_predictions = predictions.softmax(dim=1)
    return loss(softmax_predictions, one_hot_targets)


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
    scaled_variances = torch.outer(variances, sigma_sq)
    return torch.sum(
        torch.divide((predictions - targets) ** 2, scaled_variances)
        + torch.log(scaled_variances)
    )
