import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed

    def forward(self, src, tgt, src_mask, tgt_mask, query_pos=None):
        """
        Take in and process masked src and target sequences.
        """
        output = self.encode(src, src_mask)
        return self.decode(output, src_mask, tgt, tgt_mask, query_pos)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, query_pos=None):
        return self.decoder(tgt, memory, src_mask, tgt_mask, query_pos)


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, x_mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, x_mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Follow Figure 1 (left) for connections.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class PE():
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer, n, return_intermediate=False):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.size)
        self.return_intermediate = return_intermediate

    def forward(self, x, memory, src_mask, tgt_mask, query_pos=None):

        intermediate = []

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, query_pos)

            if self.return_intermediate:
                intermediate.append(self.norm(x))

        if self.norm is not None:
            x = self.norm(x)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(x)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return x


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    # TODO How to fusion the feature
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, x, memory, src_mask, tgt_mask, query_pos=None):
        """
        Follow Figure 1 (right) for connections.
        """
        m = memory
        q = k = self.with_pos_embed(x, query_pos)
        x = self.sublayer[0](x, lambda x: self.self_attn(q, k, x, tgt_mask))
        x = self.with_pos_embed(x, query_pos)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model, bias=True), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        """
        if len(query.shape) > 3:
            batch_dim = len(query.shape) - 2
            batch = query.shape[:batch_dim]
            mask_dim = batch_dim
        else:
            batch = (query.shape[0],)
            mask_dim = 1
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(dim=mask_dim)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(*batch, -1, self.h, self.d_k).transpose(-3, -2) for l, x in
                             zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(-3, -2).contiguous().view(*batch, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PointerwiseFeedforward(nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PointerwiseFeedforward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class MCG_block(nn.Module):
    def __init__(self, hidden_dim):
        super(MCG_block, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, inp, context, mask):
        context = context.unsqueeze(1)
        mask = mask.unsqueeze(-1)

        inp = self.MLP(inp)
        inp = inp * context
        inp = inp.masked_fill(mask == 0, torch.tensor(-1e9))
        context = torch.max(inp, dim=1)[0]
        return inp, context


class CG_stacked(nn.Module):
    def __init__(self, stack_num, hidden_dim):
        super(CG_stacked, self).__init__()
        self.CGs = nn.ModuleList()
        self.stack_num = stack_num
        for i in range(stack_num):
            self.CGs.append(MCG_block(hidden_dim))

    def forward(self, inp, context, mask):

        inp_, context_ = self.CGs[0](inp, context, mask)
        for i in range(1, self.stack_num):
            inp, context = self.CGs[i](inp_, context_, mask)
            inp_ = (inp_ * i + inp) / (i + 1)
            context_ = (context_ * i + context) / (i + 1)
        return inp_, context_


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, n):
    """
    Produce N identical layers.
    """
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    d_k = query.size(-1)

    # Q,K,V: [bs,h,num,dim]
    # scores: [bs,h,num1,num2]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # mask: [bs,1,1,num2] => dimension expansion

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, value=-1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model, bias=True)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class MLP_FFN(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers.children():
            nn.init.zeros_(layer.bias)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LocalSubGraphLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        """Local subgraph layer

        :param dim_in: input feat size
        :type dim_in: int
        :param dim_out: output feat size
        :type dim_out: int
        """
        super(LocalSubGraphLayer, self).__init__()
        self.mlp = MLP(dim_in, dim_in)
        self.linear_remap = nn.Linear(dim_in * 2, dim_out)

    def forward(self, x: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
        """Forward of the model

        :param x: input tensor
        :tensor (B,N,P,dim_in)
        :param invalid_mask: invalid mask for x
        :tensor invalid_mask (B,N,P)
        :return: output tensor (B,N,P,dim_out)
        :rtype: torch.Tensor
        """
        # x input -> polys * num_vectors * embedded_vector_length
        _, num_vectors, _ = x.shape
        # x mlp -> polys * num_vectors * dim_in
        x = self.mlp(x)
        # compute the masked max for each feature in the sequence

        masked_x = x.masked_fill(invalid_mask[..., None] > 0, float("-inf"))
        x_agg = masked_x.max(dim=1, keepdim=True).values
        # repeat it along the sequence length
        x_agg = x_agg.repeat(1, num_vectors, 1)
        x = torch.cat([x, x_agg], dim=-1)
        x = self.linear_remap(x)  # remap to a possibly different feature length
        return x


class LocalSubGraph(nn.Module):
    def __init__(self, num_layers: int, dim_in: int) -> None:
        """PointNet-like local subgraph - implemented as a collection of local graph layers

        :param num_layers: number of LocalSubGraphLayer
        :type num_layers: int
        :param dim_in: input, hidden, output dim for features
        :type dim_in: int
        """
        super(LocalSubGraph, self).__init__()
        assert num_layers > 0
        self.layers = nn.ModuleList()
        self.dim_in = dim_in
        for _ in range(num_layers):
            self.layers.append(LocalSubGraphLayer(dim_in, dim_in))

    def forward(self, x: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
        """Forward of the module:
        - Add positional encoding
        - Forward to layers
        - Aggregates using max
        (calculates a feature descriptor per element - reduces over points)

        :param x: input tensor (B,N,P,dim_in)
        :type x: torch.Tensor
        :param invalid_mask: invalid mask for x (B,N,P)
        :type invalid_mask: torch.Tensor
        :param pos_enc: positional_encoding for x
        :type pos_enc: torch.Tensor
        :return: output tensor (B,N,P,dim_in)
        :rtype: torch.Tensor
        """
        batch_size, polys_num, seq_len, vector_size = x.shape
        invalid_mask = ~invalid_mask
        # exclude completely invalid sequences from local subgraph to avoid NaN in weights
        x_flat = x.view(-1, seq_len, vector_size)
        invalid_mask_flat = invalid_mask.view(-1, seq_len)
        # (batch_size x (1 + M),)
        valid_polys = ~invalid_mask.all(-1).flatten()
        # valid_seq x seq_len x vector_size
        x_to_process = x_flat[valid_polys]
        mask_to_process = invalid_mask_flat[valid_polys]
        for layer in self.layers:
            x_to_process = layer(x_to_process, mask_to_process)

        # aggregate sequence features
        x_to_process = x_to_process.masked_fill(mask_to_process[..., None] > 0, float("-inf"))
        # valid_seq x vector_size
        x_to_process = torch.max(x_to_process, dim=1).values

        # restore back the batch
        x = torch.zeros_like(x_flat[:, 0])
        x[valid_polys] = x_to_process
        x = x.view(batch_size, polys_num, self.dim_in)
        return x


class MLP_3(nn.Module):
    def __init__(self, dims):
        super(MLP_3, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.LayerNorm(dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.LayerNorm(dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], dims[3])
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(f'lmlp_{i}', MLP(in_channels, in_channels))
            # in_channels = hidden_unit * 2

    def forward(self, lane):
        x = lane
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)

        x_max = torch.max(x, -2)[0]
        return x_max


def split_dim(x: torch.Tensor, split_shape: tuple, dim: int):
    if dim < 0:
        dim = len(x.shape) + dim
    return x.reshape(*x.shape[:dim], *split_shape, *x.shape[dim + 1:])
