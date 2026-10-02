import torch
import torch.nn.functional as F

def transformer_block(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor, gamma1: torch.Tensor, beta1: torch.Tensor, gamma2: torch.Tensor, beta2: torch.Tensor, mode: str, eps: float = 1e-5) -> torch.Tensor:
    def layer_norm(z:torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            z,
            normalized_shape=(z.shape[-1],),
            weight=gamma,
            bias=beta,
            eps=eps
        )
    
    if mode == "pre_norm":
        # X -> LN -> sublayer -> Residual
        n1 = layer_norm(x, gamma1, beta1)
        s1 = n1 @ W1 + b1
        x1 = x + s1

        n2 = layer_norm(x1, gamma2, beta2)
        s2 = n2 @ W2 + b2
        out = x1 + s2
    elif mode == "post_norm":
        # X -> sublayer -> residual -> LN 
        s1 = x @ W1 + b1
        x1 = layer_norm(x + s1, gamma1, beta1)

        s2 = x1 @ W2 + b2
        out = layer_norm(x1 + s2, gamma2, beta2)
    else:
        raise ValueError("mode must be 'pre_norm' or 'post_norm'")
    return out
    