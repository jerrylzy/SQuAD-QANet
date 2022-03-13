"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import qanet_layers
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDAF(nn.Module):
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
        use_char_cnn (bool): Whether to use Char-CNN
    """

    def __init__(self, char_vectors, word_vectors, hidden_size, drop_prob=0., use_char_cnn=False):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(char_vectors=char_vectors,
                                    word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    use_char_cnn=use_char_cnn)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.self_att = layers.SelfAttention(hidden_size=2 * hidden_size,
                                             num_heads=5,
                                             dropout=drop_prob)

        self.mod = layers.RNNEncoder(input_size=10 * hidden_size,
                                     hidden_size=2 * hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=0)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        # if isinstance(module, (nn.Linear, nn.Embedding)):
        #     module.weight.data.normal_(mean=0.0, std=0.02)
        #     if isinstance(module, nn.Linear) and module.bias is not None:
        #         module.bias.data.zero_()
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


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


class QANet(nn.Module):
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
        self.emb = qanet_layers.Embedding(char_vectors=char_vectors,
                                    word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    use_char_cnn=use_char_cnn)
        num_conv_layers = 4
        self.enc = qanet_layers.EncoderBlock(
            hidden_size=hidden_size,
            num_heads=8,
            dropout=drop_prob,
            kernel_size=7,
            num_conv_layers=num_conv_layers,
            base_layer_num=1,
            total_num_layers=num_conv_layers + 2)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.mod_proj = layers.Conv1dLinear(4 * hidden_size, hidden_size, bias=False) if project else None

        self.mod = qanet_layers.StackedEmbeddingEncoderBlock(
            hidden_size=hidden_size if project else 4 * hidden_size,
            num_blocks=7,
            num_heads=8,
            dropout=drop_prob,
            kernel_size=5,
            num_conv_layers=2
        )

        self.out = qanet_layers.QANetOutput(hidden_size=hidden_size if project else 4 * hidden_size, drop_prob=drop_prob)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        # if isinstance(module, (nn.Linear, nn.Embedding)):
        #     module.weight.data.normal_(mean=0.0, std=0.02)
        #     if isinstance(module, nn.Linear) and module.bias is not None:
        #         module.bias.data.zero_()
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


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
        att_emb_1 = F.dropout(self.mod(att, c_mask), self.drop_prob, self.training)
        att_emb_2 = F.dropout(self.mod(att_emb_1, c_mask), self.drop_prob, self.training)
        att_emb_3 = self.mod(att_emb_2, c_mask)

        # 2 tensors, each (batch_size, c_len)
        out = self.out(att_emb_1, att_emb_2, att_emb_3, c_mask)

        return out
