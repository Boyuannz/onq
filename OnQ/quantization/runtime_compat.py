"""Cheap startup check for the existing EfficientQAT clipping derivative."""
import torch


def require_compatible_quantization_gradients():
    """Reject changed clamp endpoint semantics without replacing the quantizer.

    Both Block-AP and native OPD use round-STE followed by ordinary clamp.
    Their reference derivative includes the two endpoints. Use fixed CPU
    tensors so this check neither allocates a GPU nor consumes training RNG.
    Evaluation needs only the forward operation and does not use this gate.
    """
    with torch.inference_mode(False), torch.enable_grad():
        codes = torch.tensor([-1., 0., 1., 3., 4.], dtype=torch.float32,
                             device='cpu', requires_grad=True)
        gradient, = torch.autograd.grad(codes.clamp(0, 3).sum(), codes)
        expected = torch.tensor([0., 1., 1., 1., 0.], device='cpu')
    if not torch.equal(gradient, expected):
        raise RuntimeError(
            f'Incompatible PyTorch {torch.__version__} quantization gradients: '
            f'clamp endpoint probe returned {gradient.tolist()}, '
            f'expected {expected.tolist()}. Use the recorded compatible training '
            'environment in OnQ/docs/ENVIRONMENT.md; do not silently '
            'change the quantizer or resume this experiment in this environment.')
