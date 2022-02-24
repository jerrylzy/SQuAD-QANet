import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import masked_softmax
from layers import HighwayEncoder


def stochastic_depth_layer_dropout(drop_prob, layer, num_layers):
    return drop_prob * layer / num_layers


class InputEmbedding(nn.Module):
    # Character embedding size limit
    CHAR_LIMIT = 16

    """Embedding layer used by BiDAF

    Char- and word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        char_vectors (torch.Tensor): Pre-trained char vectors.
        word_vectors (torch.Tensor): Pre-trained word vectors.
        emb_size (int): Size of embedding.
        drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self, char_vectors, word_vectors, hidden_size, drop_prob):
        super(InputEmbedding, self).__init__()
        self.drop_prob = drop_prob
        # vocab size, char emb dim
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False, padding_idx=0)
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, w_idx, c_idx):
        # (batch_size, seq_len, word_embed_size)
        word_emb = self.word_embed(w_idx)
        # (N batch_size, C seq_len, H char_limit, W char_embed_size)
        char_emb = self.char_embed(c_idx)
        # (batch_size, seq_len, embed_size)
        emb = torch.cat((word_emb, char_emb.view(
            *char_emb.shape[:2], -1)), dim=2)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.hwy(emb)   # (batch_size, seq_len, emb_size)

        return emb


class PositionalEncoding(nn.Module):
    """
    Fixed positional encoding layer
    """

    def __init__(self, emb_size, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # assert hidden_size % 2 == 0

        pe = torch.zeros(1, max_len, emb_size)
        i = torch.arange(0, max_len).repeat((emb_size // 2, 1)).T
        j = torch.arange(0, emb_size, 2)
        index = i * 10000 ** (-j / emb_size)
        pe[:, :, 0::2] = torch.sin(index)
        pe[:, :, 1::2] = torch.cos(index)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape = batch size, sequence size, embedding dimension
        output = self.dropout(x + self.pe[:, :x.shape[1], :])
        return output


class MultiHeadAttention(nn.Module):
    """
    Transformer Multihead Self-Attention
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        # assert hidden_size % num_heads == 0
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        self.scaled_dk = math.sqrt(self.d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # batch size, sequence size, embedding dimension
        N, S, _ = x.shape
        H = self.num_heads
        q = self.query(x).view(N, S, H, self.d_k).to(memory_format=torch.channels_last).transpose(
            1, 2)  # (N, H, S, dk)
        k = self.key(x).view(N, S, H, self.d_k).to(memory_format=torch.channels_last).transpose(
            1, 2)     # (N, H, S, dk)
        v = self.value(x).view(N, S, H, self.d_k).to(memory_format=torch.channels_last).transpose(
            1, 2)  # (N, H, S, dk)

        att = torch.matmul(q, k.transpose(2, 3)).to(
            memory_format=torch.channels_last) / self.scaled_dk  # Scaled Dot Product Attention

        att = self.dropout(self.softmax(att)).to(
            memory_format=torch.channels_last)  # (N, H, S, T)

        y = torch.matmul(att, v).to(memory_format=torch.channels_last).transpose(
            1, 2).contiguous().view(N, S, -1)
        output = self.proj(y)
        return output


class ResidualBlock(nn.Module):
    """
    Residual Block
    """

    def __init__(self, module, hidden_size, residual_dropout_p=0.1):
        super().__init__()
        self.module = module
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.residual_dropout = nn.Dropout(residual_dropout_p)

    def forward(self, x):
        # Normalize
        input = self.layer_norm(x)
        # Apply module
        output = self.residual_dropout(self.module(input))
        # Add residual connection
        output = output + x
        return output


class FeedForward(nn.Module):
    """
    Feed Forward Layer
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return F.relu(self.linear(x))


class DepthWiseSeparableConv(nn.Module):
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
                                    bias=True)
        self.point_conv = nn.Conv1d(in_channels=hidden_size,
                                    out_channels=hidden_size,
                                    kernel_size=1,
                                    bias=True)

    def forward(self, x):
        depth = self.depth_conv(x.transpose(1, 2))
        point = self.point_conv(depth).transpose(1, 2)

        return F.relu(point)


class EncoderBlock(nn.Module):
    """
    QANet Self-Attention Encoder Block Layer
    """

    def __init__(self, hidden_size, num_heads, dropout, kernel_size, num_conv_layers):
        super().__init__()

        num_layers = num_conv_layers + 2

        # Pos Encoding
        self.pe = PositionalEncoding(
            emb_size=hidden_size,
            dropout=stochastic_depth_layer_dropout(
                drop_prob=dropout,
                layer=1,
                num_layers=num_layers))

        # Conv
        self.conv = nn.Sequential(
            *[ResidualBlock(
                DepthWiseSeparableConv(hidden_size, kernel_size),
                hidden_size=hidden_size,
                residual_dropout_p=stochastic_depth_layer_dropout(
                    drop_prob=dropout,
                    layer=1 + i,
                    num_layers=num_layers)) for i in range(num_conv_layers)])

        # Attention
        self.multihead_att = ResidualBlock(
            MultiHeadAttention(hidden_size, num_heads, dropout),
            hidden_size=hidden_size,
            residual_dropout_p=stochastic_depth_layer_dropout(
                drop_prob=dropout,
                layer=num_layers + 1,
                num_layers=num_layers))

        # Feed Forward
        self.ff = ResidualBlock(FeedForward(
            hidden_size), hidden_size=hidden_size, residual_dropout_p=dropout)

    def forward(self, x):
        # Add positional encoding
        x = self.pe(x)
        # Conv
        conv = self.conv(x)
        # MultiHeadAttention
        att = self.multihead_att(conv)
        # FF
        output = self.ff(att)
        return output


class StackedEmbeddingEncoderBlock(nn.Module):
    """
    Stacked Encoder Block Layer
    """

    def __init__(self, hidden_size, num_blocks, num_heads=8, dropout=0.1, kernel_size=7, num_conv_layers=4):
        super().__init__()

        self.encoders = nn.Sequential(
            *[EncoderBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                kernel_size=kernel_size,
                num_conv_layers=num_conv_layers) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.encoders(x)


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
        self.att_linear_1 = nn.Linear(2 * hidden_size, 1)
        self.att_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, emb_1, emb_2, emb_3, mask):
        logits_1 = self.att_linear_1(torch.cat((emb_1, emb_2), dim=2))
        logits_2 = self.att_linear_2(torch.cat((emb_1, emb_3), dim=2))

        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
