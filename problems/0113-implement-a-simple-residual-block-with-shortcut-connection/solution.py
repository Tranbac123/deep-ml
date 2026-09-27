import torch

def residual_block(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """
    Implement a simple residual block with shortcut connection.
    
    Args:
        x: 1D input tensor
        w1: First weight matrix
        w2: Second weight matrix
    
    Returns:
        Output tensor after residual block processing
    """
    h1 = x @ w1
    h2 = h1 @ w2
    if h2.shape != x.shape:
        raise ValueError("Residual branch output must have the same shape as x")
    out = h2 + x
    out = torch.relu(out)
    return out