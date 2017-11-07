
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RNNEncoder(nn.Module):
    """RNN Character level encoder of the focus token"""
    def __init__(self, num_layers, input_size, hidden_size, dropout=0.0):
        self.num_layers = num_layers
        self.input_shape = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        super(RNNEncoder, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=True,
                           dropout=self.dropout)

    def init_hidden(self, token_in, batch_dim=1):
        batch = token_in.size(batch_dim)
        size = (2 * self.num_layers, batch, self.hidden_size)
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
        token_out: (seq_len x batch x hidden_size)
            Output sequence of rnn summed over directions
            for attentional decoding.
        """
        token_out, _ = self.rnn(
            token_embed, self.init_hidden(token_embed))
        # token_out (seq_len x batch x hidden_size * 2)
        seq_len, batch, _ = token_out.size()
        token_out = token_out.view(seq_len, -1, 2, self.hidden_size)
        # resp. (batch x 1 x hidden_size)
        left, right = torch.chunk(token_out, chunks=2, dim=2)
        token_out = (left + right).squeeze()
        return token_out


class ConvEncoder(nn.Module):
    """CNN Encoder of the focus token at the character level"""
    def __init__(self, in_channels, out_channels, kernel_size, output_size,
                 pooling='max', dropout=0.0):
        self.dropout = dropout
        self.pooling = pooling
        self.out_channels = out_channels
        super(ConvEncoder, self).__init__()

        self.focus_conv = nn.Conv1d(
            in_channels=in_channels,   # emb_dim
            out_channels=out_channels,  # nb_filters
            kernel_size=kernel_size)

        # !diff: token_len doesn't require dense weights
        # !diff: use some kind of pooling to abstract over l_out
        if pooling == 'rnn':
            assert divmod(output_size, 2)[1] == 0, \
                "rnn pooling requires even nb_filters"
            self.pooling_layer = nn.GRU(
                out_channels, out_channels / 2, bidirectional=True)

        self.focus_dense = nn.Linear(out_channels, output_size)

    def pool(self, token_out):
        """
        Applies some kind of pooling over the dense output after convolution.

        Parameters
        ===========
        token_out : (batch x c_out x l_out)

        Returns
        ========
        token_out : (batch x c_out)
        """
        if self.pooling == 'rnn':
            batch = token_out.size(0)
            hidden_data = token_out.data.new(2, batch, self.out_channels)
            hidden = Variable(hidden_data, requires_grad=False)
            # (l_out x batch x c_out)
            token_out = token_out.transpose(0, 2).contiguous()
            outs, _ = self.pooling_layer(token_out, hidden)
            return outs[-1]
        elif self.pooling == 'max':
            l_out = token_out.size(2)
            return F.max_pool1d(token_out, l_out).squeeze()

    def forward(self, token_embed):
        """
        Parameters
        ===========
        token_embed (seq_len x batch x emb_dim)

        Returns
        ========
        token_out : (batch x output_size)
        """
        # (batch x emb_dim x seq_len)
        token_embed = token_embed.transpose(0, 1).transpose(1, 2)
        # (batch x c_out x l_out)
        token_out = self.focus_conv(token_embed)
        token_out = F.relu(token_out)
        token_out = F.dropout(
            token_out, p=self.dropout, training=self.training)
        # (batch x c_out)
        token_out = self.pool(token_out)
        # (batch x output_size)
        token_out = self.focus_dense(token_out)
        token_out = F.dropout(
            token_out, p=self.dropout, training=self.training)
        token_out = F.relu(token_out)
        return token_out
