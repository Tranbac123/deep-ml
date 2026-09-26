import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q,K,V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    d_k = K.shape[-1]
    scores = (Q @ K.T) / (d_k ** 0.5)
    attention_weights = torch.softmax(scores, dim=-1)
    attention_output = attention_weights @ V
    return attention_output

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:

    # 1. d_head = d_model // num_heads
    T, d_model = Q.shape
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    d_head = d_model // n_heads

    # 2. reshape Q, K, V: (T,d_model) -> (T,H,d_head)-> (H,T,d_head)
    Q = Q.reshape(T, n_heads, d_head)
    Q = Q.transpose(0,1)
    K = K.reshape(T, n_heads, d_head)
    K = K.transpose(0,1)
    V = V.reshape(T, n_heads, d_head)
    V = V.transpose(0,1)

    # 3. compute attention
    head_output = [] 
    for h in range(n_heads):
        output = self_attention(Q[h],K[h],V[h])
        head_output.append(output)

    # 4. concat: (T,H,d_head) -> (T,d_model)
    concat = torch.cat(head_output, dim=-1)
    return concat
