import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import FeedForward, ResidualBlock
from util import masked_softmax, stochastic_depth_layer_dropout, get_available_devices
device, _ = get_available_devices()


class DepthWiseSeparableConv1D(nn.Module):
    """
    Depth-wise Separable Convolution
    """

    def __init__(self, hidden_size, kernel_size=7):
        super().__init__()
        self.depth_conv = nn.Conv1d(in_channels=hidden_size,
                                    out_channels=hidden_size,
                                    kernel_size=kernel_size,
                                    groups=hidden_size,
                                    padding=kernel_size // 2,
                                    bias=False,
                                    device=device)
        nn.init.xavier_uniform_(self.depth_conv.weight)
        self.point_conv = nn.Conv1d(in_channels=hidden_size,
                                    out_channels=hidden_size,
                                    kernel_size=1,
                                    bias=False,
                                    device=device)
        nn.init.kaiming_normal_(self.point_conv.weight, nonlinearity='relu')

    def forward(self, x):
        depth = self.depth_conv(x.transpose(1, 2))
        point = self.point_conv(depth).transpose(1, 2)

        return F.relu(point)


class PositionalEncoding(nn.Module):
    """
    Fixed positional encoding layer
    """

    def __init__(self, emb_size, max_len=1024):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        assert emb_size % 2 == 0

        pe = torch.zeros(1, max_len, emb_size, device=device)
        i = torch.arange(0, max_len).repeat((emb_size // 2, 1)).T
        j = torch.arange(0, emb_size, 2)
        index = i * 10000 ** (-j / emb_size)
        pe[:, :, 0::2] = torch.sin(index)
        pe[:, :, 1::2] = torch.cos(index)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape = batch size, sequence size, embedding dimension
        # output = self.dropout(x + self.pe[:, :x.shape[1], :])
        output = x + self.pe[:, :x.shape[1], :]
        return output


class MultiHeadAttention(nn.Module):
    """
    Transformer Multihead Self-Attention
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.key = nn.Linear(hidden_size, hidden_size, device=device)
        self.query = nn.Linear(hidden_size, hidden_size, device=device)
        self.value = nn.Linear(hidden_size, hidden_size, device=device)
        self.proj = nn.Linear(hidden_size, hidden_size, device=device)
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        self.scaled_dk = math.sqrt(self.d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, att_mask=None):
        # batch size, sequence size, embedding dimension
        N, S, _ = x.shape
        H = self.num_heads
        q = self.query(x).view(N, S, H, self.d_k).transpose(1, 2)  # (N, H, S, dk)
        k = self.key(x).view(N, S, H, self.d_k).transpose(1, 2)     # (N, H, S, dk)
        v = self.value(x).view(N, S, H, self.d_k).transpose(1, 2)  # (N, H, S, dk)

        att = torch.matmul(q, k.transpose(2, 3)) / self.scaled_dk  # Scaled Dot Product Attention
        if att_mask != None:
            att_mask = att_mask.view(att_mask.shape[0], 1, 1, att_mask.shape[1])
            att = att.masked_fill(att_mask == 0, -torch.inf)

        att = self.dropout(self.softmax(att))  # (N, H, S, T)

        y = torch.matmul(att, v).transpose(1, 2).contiguous().view(N, S, -1)

        return self.proj(y)


class EncoderBlock(nn.Module):
    """
    QANet Self-Attention Encoder Block Layer
    """

    def __init__(self, hidden_size, num_heads, dropout, kernel_size, num_conv_layers, base_layer_num, total_num_layers):
        super().__init__()

        self.total_num_layers = total_num_layers
        self.drop_prob = dropout

        # Pos Encoding
        self.pe = PositionalEncoding(emb_size=hidden_size)

        # Conv
        self.conv = nn.Sequential(
            *[ResidualBlock(
                DepthWiseSeparableConv1D(hidden_size, kernel_size),
                hidden_size=hidden_size,
                residual_dropout_p=stochastic_depth_layer_dropout(self.drop_prob, base_layer_num + i, self.total_num_layers)) for i in range(num_conv_layers)])

        # Attention
        self.multihead_att = ResidualBlock(
            MultiHeadAttention(hidden_size, num_heads, dropout),
            hidden_size=hidden_size,
            residual_dropout_p=stochastic_depth_layer_dropout(self.drop_prob, base_layer_num + num_conv_layers, self.total_num_layers))

        # Feed Forward
        self.ff = ResidualBlock(
            FeedForward(hidden_size),
            hidden_size=hidden_size,
            residual_dropout_p=stochastic_depth_layer_dropout(self.drop_prob, base_layer_num + num_conv_layers + 1, self.total_num_layers))

    def forward(self, x, mask=None):
        # Add positional encoding
        x = self.pe(x)
        # Conv
        conv = self.conv(x)
        # MultiHeadAttention
        att = self.multihead_att(conv, mask)
        # FF
        output = self.ff(att)
        return output


class StackedEmbeddingEncoderBlock(nn.Module):
    """
    Stacked Encoder Block Layer
    """

    def __init__(self, hidden_size, num_blocks, num_heads=8, dropout=0.1, kernel_size=7, num_conv_layers=4):
        super().__init__()
        total_num_layers = (num_conv_layers + 2) * num_blocks
        self.encoders = [EncoderBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                kernel_size=kernel_size,
                num_conv_layers=num_conv_layers,
                base_layer_num=num_block + 1,
                total_num_layers=total_num_layers) for num_block in range(num_blocks)]

    def forward(self, x, mask=None):
        for encoder in self.encoders:
            x = encoder(x, mask)
        return x


class QANetOutput(nn.Module):
    """Output layer used by QANet for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.att_linear_1 = nn.Linear(2 * hidden_size, 1, device=device)
        self.att_linear_2 = nn.Linear(2 * hidden_size, 1, device=device)

    def forward(self, emb_1, emb_2, emb_3, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(torch.cat((emb_1, emb_2), dim=2))
        logits_2 = self.att_linear_2(torch.cat((emb_1, emb_3), dim=2))

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
