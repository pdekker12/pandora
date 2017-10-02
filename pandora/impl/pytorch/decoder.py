
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pandora.utils import BOS, EOS, PAD


class LinearDecoder(nn.Module):

    """Simple Linear Decoder that outputs a probability distribution
    over the lemma vocabulary on the basis of a character-level focus
    token encoding and a token-level

    Parameters
    ===========
    input_size : int, output_size of both token and context encoders.
    output_size : int, number of lemmas in the vocabulary.
    include_context : bool, whether to use the output of the context
        encoder in the prediction of the lemma.
    """

    def __init__(self, input_size, output_size, include_context=True):
        self.output_size = output_size
        self.include_context = include_context
        super(LinearDecoder, self).__init__()

        self.decoder = nn.Linear(input_size, self.output_size)

    def forward(self, token_out, context_out, *args):
        if self.include_context:
            assert context_out is not None, \
                "LinearDecoder requires param `context_out` if the flag " + \
                "`include_context` was set to true"
            linear_in = torch.cat([token_out, context_out], 1)
        else:
            linear_in = token_out
        return F.log_softmax(self.decoder(linear_in))


class Attention(nn.Module):

    """
    Attention module.

    Parameters
    ===========
    hidden_size : int, size of both the encoder output and the attention
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, dec_out, enc_outs):
        """
        Parameters
        ===========
        dec_out : (batch x hidden_size)
            Output of the rnn decoder at the current decoding step.
        enc_outs : (seq_len x batch x hidden_size)
            Output of the encoder over the entire sequence.

        Returns
        ========
        context : (batch x hidden_size)
            Context vector combining current rnn output and the entire
            encoded sequence.
        weights : (batch x seq_len)
            Weights computed by the attentional module over the input seq.
        """
        att_proj = self.linear_in(dec_out).unsqueeze(2)
        # (seq_len x batch x hidden_size) * (batch x hidden_size x 1)
        # -> (batch x seq_len (x 1))
        weights = torch.bmm(enc_outs.transpose(0, 1), att_proj).squeeze(2)
        weights = F.softmax(weights)
        # (batch x 1 x seq_len) * (batch x seq_len x hidden_size)
        # -> (batch x 1 x hidden_size)
        weighted = torch.bmm(
            weights.unsqueeze(1), enc_outs.transpose(0, 1)
        ).squeeze(1)
        context = F.tanh(self.linear_out(torch.cat([weighted, dec_out], 1)))
        return context, weights


class AttentionalDecoder(nn.Module):

    """Character-level decoder using attention over the entire
    input sequence.

    Parameters
    ===========
    char_dict : dict used to encode token-characters into integers.
    hidden_size : int, hidden size of the encoder, decoder and
        attention modules.
    char_embed_dim : int, embedding dimension of the characters.
        It should be the same as the token_embeddings in the parent
        module to ensure that weights can be shared.
    include_context : bool, whether to use the sentential context
        encoding in the prediction of each lemma character.
    """

    def __init__(self, char_dict, hidden_size, char_embed_dim,
                 include_context=True):
        self.char_dict = char_dict
        self.char_vocab = len(char_dict)
        self.hidden_size = hidden_size
        self.char_embed_dim = char_embed_dim
        self.include_context = include_context
        super(AttentionalDecoder, self).__init__()

        self.embeddings = nn.Embedding(self.char_vocab, self.char_embed_dim)
        self.rnn = nn.LSTMCell(self.char_embed_dim, self.hidden_size)
        self.attn = Attention(self.hidden_size)
        self.proj = nn.Linear(
            self.hidden_size + self.char_embed_dim,  # include previous emb
            self.char_vocab)

    def init_hidden(self, inp, batch_dim=0):
        size = (inp.size(batch_dim), self.hidden_size)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        return h_0, c_0

    def decode(self, token_out, context_out, token_context, lemma_out):
        eos, pad = self.char_dict[EOS], self.char_dict[PAD]
        hidden, hyp = self.init_hidden(context_out), []

        # remove eos from target data
        lemma_out = lemma_out.masked_fill_(lemma_out.eq(pad), eos)[:, :-1]
        # (seq_len x batch)
        lemma_out = lemma_out.t()

        while len(hyp) < len(lemma_out):
            prev = lemma_out[len(hyp)]
            prev_emb = self.embeddings(prev)
            hidden = self.rnn(prev_emb, hidden)
            context, weights = self.attn(hidden[0], token_context)
            context = torch.cat([prev_emb, context], 1)
            log_prob = F.log_softmax(self.proj(context))
            hyp.append(log_prob)

        return torch.stack(hyp)

    def generate(self, token_out, context_out, token_context, lemma_out,
                 max_seq_len=20):
        bos = self.char_dict[BOS]
        hidden, hyp = self.init_hidden(context_out), []
        prev_data = token_out.data.new(token_out.size(0)).long()
        prev = Variable(prev_data.fill_(bos), requires_grad=False)

        while len(hyp) < max_seq_len:  # TODO: better end handling
            prev_emb = self.embeddings(prev)
            hidden = self.rnn(prev_emb, hidden)
            context, weights = self.attn(hidden[0], token_context)
            context = torch.cat([prev_emb, context], 1)
            log_prob = F.log_softmax(self.proj(context))
            prev = log_prob.max(1)[1]
            hyp.append(log_prob.data)

        return torch.stack(hyp)

    def forward(self, token_out, context_out, token_context, lemma_out,
                max_seq_len=20):
        """
        Parameters
        ===========
        token_out : (batch x hidden_size)
            Last step of encoder output. Ignored by this module.
        context_out : (batch x hidden_size)
            Output of the context encoder (sentence context information).
        token_context : (seq_len x batch x hidden_size)
            Output sequence of the encoder; target attention span.
        lemma_out : (batch x seq_len)
            Target lemma in integer format with added <eos> and <bos> symbols.
            If given it will be use for greedy training, otherwise the model
            will be fedback its own prediction at previous step for decoding.

        Returns
        ========
        hyp : (seq_len x batch x char_vocab)
            Character-level log-probabilities for the output lemma label.
        """
        if self.training:
            return self.decode(
                token_out, context_out, token_context, lemma_out)
        else:
            return self.generate(
                token_out, context_out, token_context, lemma_out, max_seq_len)
