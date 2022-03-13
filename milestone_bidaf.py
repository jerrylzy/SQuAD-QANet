"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


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
    def __init__(self, char_vectors, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob

        vocab_size, char_emb_dim = char_vectors.size(0), char_vectors.size(1)
        self.char_embed = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1) + char_vectors.size(1) * self.CHAR_LIMIT, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, w_idx, c_idx):
        word_emb = self.word_embed(w_idx)   # (batch_size, seq_len, word_embed_size)
        char_emb = self.char_embed(c_idx)   # (batch_size, seq_len, char_limit, char_embed_size)
        emb = torch.cat((word_emb, char_emb.view(*char_emb.shape[:2], -1)), dim=2)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
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
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

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
        # assert hidden_size % 2 == 0

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

    def forward(self, query, key, value, attn_mask=None):
        # batch size, sequence size, embedding dimension
        N, S, _ = query.shape
        N, T, _ = value.shape
        H = self.num_heads
        q = self.query(query).view(N, S, H, self.d_k).transpose(1, 2) # (N, H, S, dk)
        k = self.key(key).view(N, T, H, self.d_k).transpose(1, 2)     # (N, H, T, dk)
        v = self.value(value).view(N, T, H, self.d_k).transpose(1, 2) # (N, H, T, dk)

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
        assert hidden_size % num_heads == 0

        # Attention
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.multihead_att = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.residual_dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(hidden_size)

        # Feed Forward
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.residual_dropout_2 = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask):
        # Add positional encoding
        x = self.pe(x)
        # MultiHeadAttention
        att = self.residual_dropout_1(self.multihead_att(x, x, x, mask))
        att = self.layer_norm_1(att + x)
        # FF
        ff = F.relu(self.linear_1(att))
        ff = self.linear_2(ff)
        ff = self.residual_dropout_2(ff)
        output = self.layer_norm_2(ff + att)
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
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

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

        self.rnn = RNNEncoder(input_size=4 * hidden_size,
                              hidden_size=2 * hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(10 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(4 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class MilestoneBiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.
    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).
    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed char/word indices to get char/word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.
    Args:
        char_vectors (torch.Tensor): Pre-trained char vectors.
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, char_vectors, word_vectors, hidden_size, drop_prob=0.):
        super(MilestoneBiDAF, self).__init__()
        self.emb = Embedding(char_vectors=char_vectors,
                                    word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.self_att = SelfAttention(hidden_size=2 * hidden_size,
                                             num_heads=5,
                                             dropout=drop_prob)

        self.mod = RNNEncoder(input_size=10 * hidden_size,
                                     hidden_size=2 * hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # (batch_size, c_len, hidden_size)
        c_emb = self.emb(cw_idxs, cc_idxs)
        # (batch_size, q_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)

        # (batch_size, c_len, 2 * hidden_size)
        c_enc = self.enc(c_emb, c_len)
        # (batch_size, q_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        # (batch_size, c_len, 2 * hidden_size)
        self_att = self.self_att(c_enc, c_mask)

        concat_att = torch.cat((att, self_att), dim=2)

        # (batch_size, c_len, 2 * hidden_size)
        mod = self.mod(concat_att, c_len)

        # 2 tensors, each (batch_size, c_len)
        out = self.out(concat_att, mod, c_mask)

        return out
