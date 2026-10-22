"""Shared reporting-only graph cleanup; loss tensors are not passed here."""
import torch


def detach_reporting_metadata(model_output):
    """Drop graphs retained only for post-backward native GKD reporting."""
    if not isinstance(model_output, dict):
        raise TypeError('model_output reporting metadata must be a mapping')
    return {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in model_output.items()
    }
