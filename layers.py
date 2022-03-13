"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax, stochastic_depth_layer_dropout, get_available_devices
device, _ = get_available_devices()


class ResidualBlock(nn.Module):
    """
    Residual Block
    """

    def __init__(self, module, hidden_size, residual_dropout_p=0.1):
        super().__init__()
        self.module = module
        self.layer_norm = nn.LayerNorm(hidden_size, device=device)
        self.residual_dropout = nn.Dropout(residual_dropout_p)

    def forward(self, x, mask=None):
        # Normalize
        input = self.layer_norm(x)
        # Apply module
        output = self.residual_dropout(self.module(input, mask)) if mask != None else self.residual_dropout(self.module(input))
        # Add residual connection
        output = output + x
        return output


class FeedForward(nn.Module):
    """
    Feed Forward Layer
    """

    def __init__(self, hidden_size, input_size = None, output_size = None):
        super().__init__()
        self.l1 = nn.Linear(input_size if input_size != None else hidden_size, hidden_size, device=device)
        self.l2 = nn.Linear(hidden_size, output_size if output_size != None else hidden_size, device=device)
        # self.l1 = Conv1dLinear(input_size if input_size != None else hidden_size, hidden_size, use_relu=True)
        # self.l2 = Conv1dLinear(hidden_size, output_size if output_size != None else hidden_size)
        # nn.init.kaiming_normal_(self.l1.weight, nonlinearity='relu')
        # self.l1.bias.data.zero_()
        # nn.init.xavier_uniform_(self.l2.weight)
        # self.l2.bias.data.zero_()


    def forward(self, x):
        # return self.l2(self.l1(x))
        return self.l2(F.relu(self.l1(x)))


class CharCNN(nn.Module):
    """
    Character CNN
    """

    def __init__(self, char_emb_dim, hidden_size, kernel_width=5, drop_prob=0.05, char_limit=16):
        super().__init__()
        self.conv = nn.Conv2d(char_emb_dim, hidden_size, (1, kernel_width), padding=(0, kernel_width // 2), device=device) # Based on BiDAF's paper
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        self.bm = nn.BatchNorm2d(hidden_size)
        self.maxpool = nn.MaxPool2d((1, char_limit))
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        emb = self.conv(x)
        emb = self.bm(emb)
        emb = F.relu(emb)
        emb = self.maxpool(emb)
        return self.dropout(emb).squeeze(3)


class Conv1dLinear(nn.Module):
    """
    Linear layer by Conv1d
    """
    def __init__(self, input_size, output_size, use_relu=False, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(input_size, output_size, 1, bias=bias, device=device)
        self.use_relu = use_relu

        if use_relu:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        else:
            nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return F.relu(y) if self.use_relu else y


class Embedding(nn.Module):
    # Character embedding size limit
    CHAR_LIMIT = 16

    """Embedding layer used by BiDAF

    Char- and word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        char_vectors (torch.Tensor): Pre-trained char vectors.
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, char_vectors, word_vectors, hidden_size, drop_prob, use_char_cnn=False):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.num_layers = 2

        # char_emb_dim = char_vectors.size(1)
        # self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False, padding_idx=0)
        vocab_size, char_emb_dim = char_vectors.size(0), char_vectors.size(1)
        self.char_embed = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)

        # (batch_size, hidden_size, seq_len, char_limit)
        self.char_conv = CharCNN(char_emb_dim=char_emb_dim,
                                hidden_size=hidden_size,
                                kernel_width=5,
                                drop_prob=drop_prob * 0.5,
                                char_limit=self.CHAR_LIMIT) if use_char_cnn else None

        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        emb_dim = word_vectors.size(1) + (hidden_size if use_char_cnn else self.CHAR_LIMIT * char_emb_dim)

        # self.proj = Conv1dLinear(emb_dim, hidden_size, bias=False)
        self.proj = nn.Linear(emb_dim, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, w_idx, c_idx):
        word_emb = self.word_embed(w_idx)   # (batch_size, seq_len, word_embed_size)
        char_emb = self.char_embed(c_idx)   # (batch_size, seq_len, char_limit, char_embed_size)

        if self.char_conv == None:
            char_emb = char_emb.view((*char_emb.shape[:2], -1))
        else:
            # bs, sl, _, char_emb_dim = char_emb.shape
            char_emb = self.char_conv(char_emb.permute(0, 3, 1, 2)).permute(0, 2, 1) # (batch_size, seq_len, embed_size)

        emb = torch.cat((word_emb, char_emb), dim=2)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, embed_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size, device=device)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size, device=device)
                                    for _ in range(num_layers)])

        # for transform in self.transforms:
        #     nn.init.kaiming_normal_(transform.weight, nonlinearity='relu')
        #     transform.bias.data.zero_()

        # for gate in self.gates:
        #     nn.init.kaiming_normal_(gate.weight, nonlinearity='sigmoid')
        #     gate.bias.data.zero_()

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class PositionalEncoding(nn.Module):
    """
    Fixed positional encoding layer
    """
    def __init__(self, hidden_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert hidden_size % 2 == 0

        pe = torch.zeros(1, max_len, hidden_size)
        i = torch.arange(0, max_len).repeat((hidden_size // 2, 1)).T
        j = torch.arange(0, hidden_size, 2)
        index = i * 10000 ** (-j / hidden_size)
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
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0

        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        self.scaled_dk = math.sqrt(self.d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # batch size, sequence size, embedding dimension
        N, S, _ = x.shape
        H = self.num_heads
        q = self.query(x).view(N, S, H, self.d_k).transpose(1, 2) # (N, H, S, dk)
        k = self.key(x).view(N, S, H, self.d_k).transpose(1, 2)     # (N, H, T, dk)
        v = self.value(x).view(N, S, H, self.d_k).transpose(1, 2) # (N, H, T, dk)

        attn = torch.matmul(q, k.transpose(2, 3)) / self.scaled_dk # Scaled Dot Product Attention

        if attn_mask != None:
            attn_mask = attn_mask.view(N, 1, 1, S)
            attn = attn.masked_fill(attn_mask == 0, -math.inf)

        attn = self.dropout(self.softmax(attn)) # (N, H, S, T)

        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(N, S, -1)
        output = self.proj(y)
        return output


class SelfAttention(nn.Module):
    """
    BiDAF Self-Attention Layer
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()

        # Attention
        self.pe = PositionalEncoding(hidden_size, 0)
        # self.layer_norm_0 = nn.LayerNorm(hidden_size)
        self.multihead_att = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.residual_dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(hidden_size)

        # Feed Forward
        self.ff = FeedForward(hidden_size)
        self.residual_dropout_2 = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):

        # x = self.layer_norm_0(x)

        # Add positional encoding
        x = self.pe(x)

        # MultiHeadAttention
        attn = self.multihead_att(x, mask)
        attn = self.residual_dropout_1(attn)
        attn = self.layer_norm_1(attn + x)

        # FF
        ff = self.ff(attn)
        ff = self.residual_dropout_2(ff)
        output = self.layer_norm_2(ff + attn)
        return output


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1, device=device))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1, device=device))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size, device=device))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1, device=device))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(10 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(4 * hidden_size, 1)
        self.dropout_1 = nn.Dropout(drop_prob)

        self.rnn = RNNEncoder(input_size=4 * hidden_size,
                              hidden_size=2 * hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(10 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(4 * hidden_size, 1)
        self.dropout_2 = nn.Dropout(drop_prob)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.dropout_1(self.att_linear_1(att) + self.mod_linear_1(mod))
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.dropout_2(self.att_linear_2(att) + self.mod_linear_2(mod_2))

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
