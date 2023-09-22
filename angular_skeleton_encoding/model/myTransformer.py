#  Copyright (c) 2023. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

"""
@Project: 2023-GCN-action-recognize-tutorial
@FileName: myTransformer.py
@Description: Transformer Encoder for Network Structure adaptation
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2023/8/5 21:43 at PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # B Hn L Hd
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # B Hn L Hd
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # B Hn L Hd

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # B Hn L Hd * B Hn Hd L -> B Hn L L

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # B Hn L L'
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)  # B Hn L L' * B Hn L Hd -> B Hn L Hd
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # B L Hn*Hd
        out = self.out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        assert (
                d_model % num_heads == 0
        ), "Embedding size needs to be divisible by heads"
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # self.norm1 = nn.BatchNorm1d(d_model)
        # self.norm2 = nn.BatchNorm1d(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head self-attention
        attn_output = self.self_attn(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        residual = x + attn_output
        residual = residual.view(-1, attn_output.shape[2])
        out1 = self.norm1(residual)

        # Feed-forward neural network
        ff_output = self.feed_forward(out1)
        ff_output = self.dropout2(ff_output)
        out2 = self.norm2(out1 + ff_output)
        out2 = out2.view(x.shape[0], -1, x.shape[2])
        return out2


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


if __name__ == '__main__':
    N, c1, T, V = 1, 8, 3, 17
    trans = TransformerEncoder(
        num_layers=3,
        d_model=c1,
        num_heads=4,
        d_ff=c1 * 2
    )
    x = torch.randn(size=(N, c1, T, V))
    x = x.permute(0, 2, 3, 1).contiguous().view(N, T * V, c1)
    x = trans(x)
    x = x.view(N, T, V, c1).permute(0, 3, 1, 2).contiguous()
    print(x.shape)
