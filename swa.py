import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim 必须能整除 num_heads"

        # 线性映射
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        """
        B, L, C = x.shape

        # Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 用padding方便取窗口
        pad_w = self.window_size
        k = F.pad(k, (0, 0, pad_w, pad_w))  # (B, H, L+2w, D)
        v = F.pad(v, (0, 0, pad_w, pad_w))

        outputs = []
        for i in range(L):
            # 取[i, i+2w+1) 的 keys/values
            k_win = k[:, :, i:i+2*pad_w+1, :]   # (B, H, 2w+1, D)
            v_win = v[:, :, i:i+2*pad_w+1, :]
            q_i = q[:, :, i:i+1, :]             # (B, H, 1, D)

            attn_scores = torch.matmul(q_i, k_win.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_probs = F.softmax(attn_scores, dim=-1)  # (B, H, 1, 2w+1)

            out_i = torch.matmul(attn_probs, v_win)  # (B, H, 1, D)
            outputs.append(out_i)

        out = torch.cat(outputs, dim=2)  # (B, H, L, D)
        out = out.transpose(1, 2).reshape(B, L, C)  # (B, L, C)
        return self.out_proj(out)


# 测试
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, C = 2, 16, 32
    num_heads = 4
    window_size = 2
    x = torch.randn(B, L, C)

    attn = SlidingWindowAttention(embed_dim=C, num_heads=num_heads, window_size=window_size)
    y = attn(x)
    print(y.shape)  # (2, 16, 32)
