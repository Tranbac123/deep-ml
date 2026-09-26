import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d_k = Q.shape[1]
    score = (Q @ K.transpose(-2,-1)) / (d_k ** 0.5)
    score_mask = score + mask
    attention_weights = torch.softmax(score_mask, dim=-1)
    output = attention_weights @ V
    return output
