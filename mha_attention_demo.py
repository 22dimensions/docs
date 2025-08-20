import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, input):
        batch_size = input.size(0)
        seq_len = input.size(1)

        # Linear transformations
        query = self.query_linear(input)
        key = self.key_linear(input)
        value = self.value_linear(input)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.d_k)
        key = key.view(batch_size, seq_len, self.num_heads, self.d_k)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose dimensions for matrix multiplication
        query = query.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        key = key.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        value = value.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        attention_weights = nn.Softmax(dim=-1)(scores)
        print(attention_weights.shape)
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len, d_k]

        # Concatenate and reshape
        # [batch_size, seq_len, num_heads, d_k] -> [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Linear transformation for final output
        attention_output = self.output_linear(attention_output)

        return attention_output

# 使用示例
d_model = 256  # 输入维度
num_heads = 8  # 注意力头数

# 创建Multi-Head Attention层
attention = MultiHeadAttention(d_model, num_heads)

# 创建输入张量
batch_size = 4
seq_len = 10
input = torch.randn(batch_size, seq_len, d_model)

# 前向传播
output = attention(input)

print("输入维度:", input.shape)
print("输出维度:", output.shape)
