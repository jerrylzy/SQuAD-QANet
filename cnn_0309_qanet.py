"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        char_emb_dim = char_vectors.size(1)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False, padding_idx=0)

        # (batch_size, hidden_size, seq_len, char_limit)
        self.char_conv = CharCNN(char_emb_dim=char_emb_dim,
                                hidden_size=hidden_size,
                                kernel_width=5,
                                drop_prob=drop_prob * 0.5,
                                char_limit=self.CHAR_LIMIT) if use_char_cnn else None

        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.word_dropout = nn.Dropout(drop_prob)
        emb_dim = word_vectors.size(1) + (hidden_size if use_char_cnn else self.CHAR_LIMIT * char_emb_dim)

        # self.proj = Conv1dLinear(emb_dim, hidden_size, bias=False)
        self.proj = nn.Linear(emb_dim, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, w_idx, c_idx):
        word_emb = self.word_dropout(self.word_embed(w_idx))   # (batch_size, seq_len, word_embed_size)
        char_emb = self.char_embed(c_idx)   # (batch_size, seq_len, char_limit, char_embed_size)

        if self.char_conv == None:
            char_emb = char_emb.view((*char_emb.shape[:2], -1))
            F.dropout(char_emb, self.drop_prob * 0.5, self.training)
        else:
            # bs, sl, _, char_emb_dim = char_emb.shape
            char_emb = self.char_conv(char_emb.permute(0, 3, 1, 2)).permute(0, 2, 1) # (batch_size, seq_len, embed_size)

        emb = torch.cat((word_emb, char_emb), dim=2)   # (batch_size, seq_len, embed_size)
        # emb = F.dropout(emb, stochastic_depth_layer_dropout(self.drop_prob, 1, self.num_layers), self.training)
        # emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, embed_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        # emb = F.dropout(emb, self.drop_prob, self.training)

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

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class PositionalEncoding(nn.Module):
    """
    Fixed positional encoding layer
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Transformer Multihead Self-Attention
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0

        self.ma = nn.MultiheadAttention(hidden_size, num_heads, dropout, batch_first=True, device=device)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.ma(x, x, x, key_padding_mask = attn_mask.int())
        return attn_output


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


class DepthWiseSeparableConv1d(nn.Module):
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
        self.point_conv = nn.Conv1d(in_channels=hidden_size,
                                    out_channels=hidden_size,
                                    kernel_size=1,
                                    bias=True,
                                    device=device)
        nn.init.xavier_uniform_(self.depth_conv.weight)
        nn.init.kaiming_normal_(self.point_conv.weight, nonlinearity='relu')
        nn.init.constant_(self.point_conv.bias, 0.0)

    def forward(self, x):
        depth = self.depth_conv(x.transpose(1, 2))
        point = self.point_conv(depth).transpose(1, 2)

        return F.relu(point)


class EncoderBlock(nn.Module):
    """
    QANet Self-Attention Encoder Block Layer
    """

    def __init__(self, hidden_size, num_heads, dropout, kernel_size, num_conv_layers, base_layer_num, total_num_layers):
        super().__init__()

        self.total_num_layers = total_num_layers
        self.drop_prob = dropout

        # Pos Encoding
        self.pe = PositionalEncoding(hidden_size, dropout)

        # Conv
        self.conv = nn.Sequential(
            *[ResidualBlock(
                DepthWiseSeparableConv1d(hidden_size, kernel_size),
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
        self.encoders = nn.ModuleList([EncoderBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                kernel_size=kernel_size,
                num_conv_layers=num_conv_layers,
                base_layer_num=num_block + 1,
                total_num_layers=total_num_layers) for num_block in range(num_blocks)])

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

    def __init__(self, hidden_size, drop_prob):
        super().__init__()
        self.att_linear_1 = nn.Linear(2 * hidden_size, 1, device=device)
        self.dropout_1 = nn.Dropout(drop_prob)
        self.att_linear_2 = nn.Linear(2 * hidden_size, 1, device=device)
        self.dropout_2 = nn.Dropout(drop_prob)

    def forward(self, emb_1, emb_2, emb_3, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.dropout_1(self.att_linear_1(torch.cat((emb_1, emb_2), dim=2)))
        logits_2 = self.dropout_2(self.att_linear_2(torch.cat((emb_1, emb_3), dim=2)))

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class CNNQANet0309(nn.Module):
    # Character embedding size limit
    CHAR_LIMIT = 16

    """QANet model for SQuAD.
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
        use_char_cnn (bool): Whether to use Char-CNN
    """

    def __init__(self, char_vectors, word_vectors, hidden_size=128, drop_prob=0., project=False, use_char_cnn=True):
        super().__init__()
        self.drop_prob = drop_prob

        # Dimension of the embedding layer output.
        self.emb = Embedding(char_vectors=char_vectors,
                                    word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    use_char_cnn=use_char_cnn)
        num_conv_layers = 4
        self.enc = EncoderBlock(
            hidden_size=hidden_size,
            num_heads=8,
            dropout=drop_prob,
            kernel_size=7,
            num_conv_layers=num_conv_layers,
            base_layer_num=1,
            total_num_layers=num_conv_layers + 2)

        self.att = BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.mod_proj = nn.Linear(4 * hidden_size, hidden_size, bias=False) if project else None

        self.mod = StackedEmbeddingEncoderBlock(
            hidden_size=hidden_size if project else 4 * hidden_size,
            num_blocks=7,
            num_heads=8,
            dropout=drop_prob,
            kernel_size=5,
            num_conv_layers=2
        )

        self.out = QANetOutput(hidden_size=hidden_size if project else 4 * hidden_size, drop_prob=drop_prob)

    #     self.apply(self._init_weights)


    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)


    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        # c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # (batch_size, c_len, emb_size)
        c_emb = self.emb(cw_idxs, cc_idxs)
        # (batch_size, q_len, emb_size)
        q_emb = self.emb(qw_idxs, qc_idxs)

        c_enc = self.enc(c_emb, c_mask)    # (batch_size, c_len, hidden_size)
        q_enc = self.enc(q_emb, q_mask)    # (batch_size, q_len, hidden_size)

        att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 4 * hidden_size)

        att = self.mod_proj(att) if self.mod_proj != None else att

        # stackd encoder blocks share weights among its three repetitions
        att_emb_1 = self.mod(att, c_mask)
        att_emb_2 = self.mod(att_emb_1, c_mask)
        att_emb_3 = self.mod(att_emb_2, c_mask)

        # 2 tensors, each (batch_size, c_len)
        out = self.out(att_emb_1, att_emb_2, att_emb_3, c_mask)

        return out
