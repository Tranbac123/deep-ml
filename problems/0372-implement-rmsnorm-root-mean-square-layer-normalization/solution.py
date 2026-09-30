import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Apply RMSNorm to the input tensor.

    Parameters:
        x   : torch.Tensor of shape (batch_size, features)
        g   : torch.Tensor of shape (features,) - gain parameter
        eps : float - small constant for numerical stability

    Returns:
        torch.Tensor of same shape as x
    """
    rms_x = torch.sqrt(
        torch.mean(x**2, dim=-1, keepdim=True) + eps
    )
    x_hat = x/rms_x
    y = g * x_hat
    return y