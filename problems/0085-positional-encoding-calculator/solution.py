import torch

def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    Args:
        position: sequence length (number of positions)
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model) with dtype float16,
        or -1 if position == 0 or d_model <= 0.
    """
    # 1. invalid input
    if position <= 0 or d_model <= 0:
        return -1
    # 2. positions: (position, 1)
    pos = torch.arange(position, dtype=torch.float32).unsqueeze(1)

    # 3. even dimensions: 0,2,4,...
    dims = torch.arange(0, d_model, 2, dtype=torch.float32)

    # 4. frequncy / demonstration
    div_term = 10000 ** (dims/d_model)

    # 5. angles via broadcasting
    angles = pos / div_term

    # 6. output
    pe = torch.zeros(
        (position, d_model),
        dtype=torch.float32
    )
    # 7. even -> sin, old -> cos
    pe[:,0::2] = torch.sin(angles)
    pe[:,1::2] = torch.cos(angles)

    # 8.required dtype
    return pe.to(torch.float16)

