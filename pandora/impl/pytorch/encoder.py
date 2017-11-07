
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pandora.impl.pytorch import utils


class RNNEncoder(nn.Module):
    """RNN Character level encoder of the focus token"""
    def __init__(self, num_layers, input_size, hidden_size, dropout=0.0,
                 merge_mode='concat'):
        self.num_layers = num_layers
        self.input_shape = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.merge_mode = merge_mode
        super(RNNEncoder, self).__init__()

        self.rnn_hidden_size = self.hidden_size
        if self.merge_mode == 'concat':
            hidden_size, rest = divmod(self.hidden_size, 2)
            if rest > 0:
                raise ValueError("'concat' merge_mode needs even hidden_size")
            self.rnn_hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=self.rnn_hidden_size,
                           num_layers=num_layers,
                           bidirectional=True,
                           dropout=self.dropout)

        self.init()

    def init(self):
        # rnn
        utils.init_rnn(self.rnn)

    def init_hidden(self, token_in, batch_dim=1):
        batch = token_in.size(batch_dim)
        size = (2 * self.num_layers, batch, self.rnn_hidden_size)
        h_0 = Variable(token_in.data.new(*size).zero_(), requires_grad=False)
        c_0 = Variable(token_in.data.new(*size).zero_(), requires_grad=False)
        return h_0, c_0

    def forward(self, token_embed):
        """
        Parameters
        ===========
        token_embed: (seq_len x batch x emb_dim)

        Returns:
        =========
        token_out : (batch x hidden_size), summary vector for each batch,
            consisting of the last hidden activation of the rnn
        token_context: (seq_len x batch x hidden_size), output of the rnn
            over the entire sequence
        """
        output, _ = self.rnn(
            token_embed, self.init_hidden(token_embed))

        if self.merge_mode == 'sum':
            # (seq_len x batch x hidden_size * 2)
            seq_len, batch, _ = output.size()
            # expose bidirectional hidden (seq_len x batch x 2 x hidden)
            output = output.view(seq_len, -1, 2, self.rnn_hidden_size)
            # (seq_len x batch x 1 x hidden_size)
            left, right = torch.chunk(output, chunks=2, dim=2)
            # (seq_len x batch x hidden_size)
            output = (left + right).squeeze(2)

        token_out, token_context = output[-1], output

        return token_out, token_context


class BottleEncoder(nn.Module):
    """
    Implements a linear projection from (seq_len x batch x inp_size) to
    (batch x output_size) applying one of three different pooling operations.
    If `flatten`, it applies the linear transformation on the flattend input
    (seq_len * inp_size) to output_size. Flatten requires a known fixed input
    length. If `max` it does max pooling over the seq_len dimension prior
    to the linear transformation. If `rnn`, it runs a bidirection GRU over
    the seq_len and applies the linear transformation on the concatenated
    summary vectors of both directions.
    """
    def __init__(self, inp_size, output_size, seq_len=None, pooling='flatten',
                 dropout=0.0):
        self.pooling = pooling
        self.dropout = dropout
        self.inp_size = inp_size
        super(BottleEncoder, self).__init__()

        if self.pooling == 'flatten':
            assert seq_len, "Flatten requires knowing the seq_len"
            # no pooling, do flattening instead
            self.dense = nn.Linear(seq_len * inp_size, output_size)

        elif self.pooling == 'rnn':
            assert divmod(inp_size, 2)[1] == 0, \
                "rnn pooling requires even input size"
            self.pooling_layer = nn.GRU(
                inp_size, inp_size // 2, bidirectional=True)

            self.dense = nn.Linear(inp_size, output_size)

        elif self.pooling == 'max':
            self.dense = nn.Linear(inp_size, output_size)

    def init(self):
        if self.pooling == 'rnn':
            # rnn
            utils.init_rnn(self.pooling_layer)

        # linear
        utils.init_linear(self.dense)

    def forward(self, inp):
        """
        Parameters
        ===========
        inp : (batch x inp_size x seq_len)

        Returns
        ========
        output : (batch x output_size)
        """
        if self.pooling == 'flatten':
            inp = inp.view(inp.size(0), -1)

        elif self.pooling == 'rnn':
            # initial hidden
            batch, num_dirs, hid_size = inp.size(0), 2, self.inp_size // 2
            hidden = Variable(
                inp.data.new(num_dirs, batch, hid_size).zero_(),
                requires_grad=False)
            # (batch x inp_size x seq_len) -> (seq_len x batch x inp_size)
            inp = inp.transpose(1, 2).transpose(0, 1).contiguous()
            # (seq_len x batch x inp_size)
            inp, _ = self.pooling_layer(inp, hidden)
            # (batch x inp_size)
            inp = inp[-1]

        elif self.pooling == 'max':
            # (batch x inp_size x seq_len) -> (batch x inp_size)
            inp, _ = inp.max(2)

        inp = F.dropout(inp, p=self.dropout, training=self.training)

        return self.dense(inp)


def get_conv_output_length(inp_len, kernel_size,
                           padding=0, dilation=1, stride=1):
    """
    compute length of the convolutional output sequence (l_out)
    l_out = floor(
      (l_in + 2 ∗ padding − dilation ∗ (kernel_size − 1) − 1)
      /
      stride + 1)
    """
    return math.floor(
        (inp_len + 2 * padding - dilation * (kernel_size - 1) - 1)
        /
        stride + 1)


class ConvEncoder(nn.Module):
    """CNN Encoder of the focus token at the character level"""
    def __init__(self, in_channels, out_channels, kernel_size, output_size,
                 token_len, dropout=0.0, pooling='flatten'):
        self.dropout = dropout
        self.pooling = pooling
        self.out_channels = out_channels
        super(ConvEncoder, self).__init__()

        self.focus_conv = nn.Conv1d(
            in_channels=in_channels,    # emb_dim
            out_channels=out_channels,  # nb_filters
            kernel_size=kernel_size)

        seq_len = None
        if pooling == 'flatten':
            seq_len = get_conv_output_length(token_len, kernel_size)

        self.focus_dense = BottleEncoder(
            out_channels, output_size, seq_len=seq_len, pooling=pooling,
            dropout=dropout)

        self.init()

    def init(self):
        # conv
        utils.init_conv(self.focus_conv)

    def forward(self, token_embed):
        """
        Parameters
        ===========
        token_embed (seq_len x batch x emb_dim)

        Returns
        ========
        token_out : (batch x output_size)
        token_context : (conv_seq_len x batch x channel_out)
        """
        # (batch x emb_dim x seq_len)
        token_embed = token_embed.transpose(0, 1).transpose(1, 2)
        # (batch x channel_out x conv_seq_len)
        token_context = self.focus_conv(token_embed)
        token_context = F.relu(token_context)
        # (batch x output_size)
        token_out = self.focus_dense(token_context)
        token_out = F.dropout(
            token_out, p=self.dropout, training=self.training)
        token_out = F.relu(token_out)

        return token_out, token_context.transpose(0, 1).transpose(0, 2)
