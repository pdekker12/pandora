
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pandora.utils import BOS, EOS, PAD
from pandora.impl.pytorch import utils
from pandora.impl.pytorch.beam import Beam


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

    def __init__(self, input_size, output_size, include_context=True,
                 dropout=0.0):
        self.output_size = output_size
        self.include_context = include_context
        self.dropout = dropout
        super(LinearDecoder, self).__init__()

        self.decoder = nn.Linear(input_size, self.output_size)

        self.init()

    def init(self):
        # linear
        utils.init_linear(self.decoder)

    def forward(self, token_out, context_out, *args):
        if self.include_context:
            assert context_out is not None, \
                "LinearDecoder requires param `context_out` if the flag " + \
                "`include_context` was set to true"
            linear_in = torch.cat([token_out, context_out], 1)
        else:
            linear_in = token_out

        linear_out = self.decoder(linear_in)

        linear_out = F.dropout(
            linear_out, p=self.dropout, training=self.training)

        return F.log_softmax(linear_out)


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

        self.init()

    def init(self):
        utils.init_linear(self.linear_in)
        utils.init_linear(self.linear_out)

    def forward(self, dec_outs, enc_outs):
        """
        Parameters
        ===========
        dec_outs : (out_seq_len x batch x hidden_size)
            Output of the rnn decoder.
        enc_outs : (inp_seq_len x batch x hidden_size)
            Output of the encoder over the entire sequence.

        Returns
        ========
        context : (out_seq_len x batch x hidden_size)
            Context vector combining current rnn output and the entire
            encoded sequence.
        weights : (out_seq_len x batch x inp_seq_len)
            Weights computed by the attentional module over the input seq.
        """
        out_seq, batch, hidden_size = dec_outs.size()
        # (out_seq_len x batch x hidden_size)
        att_proj = self.linear_in(
            dec_outs.view(out_seq * batch, -1)
        ).view(out_seq, batch, -1)
        # (batch x out_seq_len x hidden) * (batch x hidden x inp_seq_len)
        # -> (batch x out_seq_len x inp_seq_len)
        weights = torch.bmm(
            att_proj.transpose(0, 1),
            enc_outs.transpose(0, 1).transpose(1, 2))
        # apply softmax
        weights = F.softmax(
            weights.view(batch * out_seq, -1)
        ).view(batch, out_seq, -1)
        # (batch x out_seq_len x inp_seq_len) * (batch x inp_seq_len x hidden)
        # -> (batch x out_seq_len x hidden_size)
        weighted = torch.bmm(weights, enc_outs.transpose(0, 1))
        # (out_seq_len x batch x hidden * 2)
        combined = torch.cat([weighted.transpose(0, 1), dec_outs], 2)
        # (out_seq_len x batch x hidden)
        combined = self.linear_out(
            combined.view(out_seq * batch, -1)
        ).view(out_seq, batch, -1)

        context = F.tanh(combined)

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

        self.rnn = nn.GRU(self.char_embed_dim, self.hidden_size)
        self.attn = Attention(self.hidden_size)
        self.proj = nn.Linear(
            self.hidden_size +      # recurrent hidden
            self.char_embed_dim,    # include previous emb
            self.char_vocab)

        self.init()

    def init(self):
        # embeddings
        utils.init_embeddings(self.embeddings)
        # linear
        utils.init_linear(self.proj)
        # rnn
        utils.init_rnn(self.rnn)

    def decode(self, token_out, context_out, token_context, lemma_out):
        eos, pad = self.char_dict[EOS], self.char_dict[PAD]
        # use last encoder state as hidden: (1 x batch x hidden)
        hidden = token_context[-1].unsqueeze(0)
        # append token context as extra input sequence step
        # token_context: (inp_seq + 1 x batch x hidden)
        token_context = torch.cat([token_context, context_out.unsqueeze(0)])

        # remove eos from target data
        lemma_out = lemma_out.masked_fill_(lemma_out.eq(pad), eos)[:, :-1]
        # (seq_len x batch)
        lemma_out = lemma_out.t()
        out_seq, batch = lemma_out.size()
        # (seq_len x batch x emb_dim)
        lemma_out = self.embeddings(lemma_out)

        # run decoder rnn
        dec_outs, _ = self.rnn(lemma_out, hidden)
        # compute attention
        context, weights = self.attn(dec_outs, token_context)
        # (out_seq_len x batch x hidden + emb_dim)
        output = torch.cat([lemma_out, context], 2)
        # (out_seq_len * batch x vocab)
        output = self.proj(output.view(out_seq * batch, -1))
        # (out_seq_len x batch x vocab)
        output = F.log_softmax(output).view(out_seq, batch, -1)

        return output

    def generate(self, token_out, context_out, token_context, lemma_out,
                 max_seq_len=20):
        bos = self.char_dict[BOS]
        # use last encoder state as hidden: (1 x batch x hidden)
        hidden, hyp = token_context[-1].unsqueeze(0), []
        batch = token_out.size(0)
        prev_data = token_out.data.new([bos]).expand(batch).long()
        prev = Variable(prev_data, requires_grad=False)
        # append token context as extra input sequence step
        # token_context: (inp_seq + 1 x batch x hidden)
        token_context = torch.cat([token_context, context_out.unsqueeze(0)])

        while len(hyp) < max_seq_len:  # TODO: better finishing handling
            # prev_emb: (1 x batch x emb_dim)
            prev_emb = self.embeddings(prev).unsqueeze(0)
            # output: (1 x batch x hidden)
            output, hidden = self.rnn(prev_emb, hidden)
            # context: (1 x batch x hidden)
            context, weights = self.attn(hidden, token_context)
            # context: (1 x batch x hidden + emb_dim)
            # -> (batch x hidden + emb_dim)
            context = torch.cat([prev_emb, context], 2).squeeze(0)
            # (batch x vocab)
            log_prob = F.log_softmax(self.proj(context))
            _, prev = log_prob.max(1)
            hyp.append(log_prob.data)

        return torch.stack(hyp)

    def forward(self, token_out, context_out, token_context, lemma_out,
                max_seq_len=20):
        """
        Parameters
        ===========
        token_out : (batch x hidden_size)
            Last step of encoder output. Ignored by this module.
        context_out : (batch x nb_dense_dims)
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

    def beam(self, token_out, context_out, token_context, lemma_out, bwidth=5,
             max_seq_len=20):
        bos, eos, output = self.char_dict[BOS], self.char_dict[EOS], []
        pad = self.char_dict[PAD]
        # use last encoder state as hidden: (1 x batch x hidden)
        hidden = token_context[-1].unsqueeze(0)
        # append token context as extra input sequence step
        # token_context: (inp_seq + 1 x batch x hidden)
        token_context = torch.cat([token_context, context_out.unsqueeze(0)])
        # prev
        prev = token_out.data.new([bos]).expand(bwidth).long()
        prev = Variable(prev, requires_grad=False)

        # make batch first
        batch = token_out.size(0)
        for hidden, token_context in zip(
                hidden.chunk(batch, 1), token_context.chunk(batch, 1)):
            # create beam
            beam = Beam(bwidth, prev, eos=eos)
            # broadcast single item batch to bwidth
            hidden = utils.broadcast(hidden, bwidth, 1)
            token_context = utils.broadcast(token_context, bwidth, 1)

            while beam.active and len(beam) < max_seq_len:
                # (1 x bwidth x emb_dim)
                prev_emb = self.embeddings(prev).unsqueeze(0)
                # output: (1 x bwidth x hidden)
                _, hidden = self.rnn(prev_emb, hidden)
                # context: (1 x bwidth x hidden)
                context, _ = self.attn(hidden, token_context)
                # context: (1 x bwidth x hidden + emb_dim)
                # -> (bwidth x hidden + emb_dim)
                context = torch.cat([prev_emb, context], 2).squeeze(0)
                # (bwidth x vocab)
                log_prob = F.log_softmax(self.proj(context))
                beam.advance(log_prob.data)

                # repackage
                hidden = utils.multiple_index_select(
                    hidden, beam.get_source_beam().unsqueeze(0))

            _, hyp = beam.decode(n=1)

            # remove bwidth dim
            hyp = hyp[0]
            # pad output sequences to same output length
            hyp = hyp + [pad] * (max_seq_len - len(hyp))
            output.append(torch.LongTensor(hyp))

        output = torch.stack(output).t()

        return output
