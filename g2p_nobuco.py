import pandas as pd
import numpy as np
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import torchtext.data as data
import Levenshtein  # https://github.com/ztane/python-Levenshtein/
from typing import Optional
import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer
import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
import keras

parser = {
    'data_path': '',
    'epochs': 50,
    'batch_size': 3700,
    'max_len': 32,  # max length of grapheme/phoneme sequences
    'beam_size': 10,  # size of beam for beam-search
    'd_embed': 500,  # embedding dimension
    'd_hidden': 500,  # hidden dimension
    'attention': True,  # use attention or not
    'log_every': 100,  # number of iterations to log and validate training
    'lr': 0.007,  # initial learning rate
    'lr_decay': 0.5,  # decay lr when not observing improvement in val_loss
    'lr_min': 1e-5,  # stop when lr is too low
    'n_bad_loss': 5,  # number of bad val_loss before decaying
    'clip': 2.3,  # clip gradient, to avoid exploding gradient
    'cuda': True,  # using gpu or not
    'seed': 5,  # initial seed
    'intermediate_path': '',  # path to save models
}
args = argparse.Namespace(**parser)

args.cuda = args.cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

class Encoder(nn.Module):

    def __init__(self, vocab_size, d_embed, d_hidden):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.lstm = nn.LSTMCell(d_embed, d_hidden)
        self.d_hidden = d_hidden

    def forward(self, x_seq):
        o = []
        e_seq = self.embedding(x_seq)  # seq x batch x dim
        tt = torch
        # create initial hidden state and initial cell state
        h = tt.zeros([e_seq.size(1), self.d_hidden], dtype=torch.float32)
        c = tt.zeros([e_seq.size(1), self.d_hidden], dtype=torch.float32)

        for e in torch.chunk(e_seq, e_seq.size(0).item(), 0):
            e = e.squeeze(0)
            h, c = self.lstm(e, (h, c))
            o.append(h)
        return tt.stack(o, 1), h, c
    
class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, pad=1, bos=2, eos=3, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0,
                                                     True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId // num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True
        return self.done

    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
        return hyp[::-1]

def phoneme_error_rate(p_seq1, p_seq2):
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return Levenshtein.distance(''.join(c_seq1),
                                ''.join(c_seq2)) / len(c_seq2)

def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay

def validate(val_iter, model, criterion):
    model.eval()
    val_loss = 0
    val_iter.init_epoch()
    for batch in val_iter:
        batch.phoneme = batch.phoneme.to(device)
        batch.grapheme = batch.grapheme.to(device)
        output, _, __ = model(batch.grapheme, batch.phoneme[:-1])
        target = batch.phoneme[1:]
        loss = criterion(output.squeeze(1), target.squeeze(1))
        val_loss += loss.item() * batch.batch_size
        return val_loss / len(val_iter.dataset)
    
def test(test_iter, model, criterion):
    model.eval()
    test_iter.init_epoch()
    test_per = test_wer = 0
    for batch in test_iter:
        batch.phoneme = batch.phoneme.to(device)
        batch.grapheme = batch.grapheme.to(device)
        output = model(batch.grapheme).data.tolist()
        target = batch.phoneme[1:].squeeze(1).data.tolist()
        per = phoneme_error_rate(output, target)
        wer = int(output != target)
        test_per += per
        test_wer += wer
    test_per = test_per / len(test_iter.dataset) * 100
    test_wer = test_wer / len(test_iter.dataset) * 100
    print("Phoneme error rate (PER): {:.2f}\nWord error rate (WER): {:.2f}"
          .format(test_per, test_wer))
    
g_field = data.Field(init_token='<s>',
                     tokenize=(lambda x: list(x.split('(')[0])[::-1]))
p_field = data.Field(init_token='<os>', eos_token='</os>',
                     tokenize=(lambda x: x.split('#')[0].split()))

class CMUDict(data.Dataset):

    def __init__(self, data_lines, g_field, p_field):
        fields = [('grapheme', g_field), ('phoneme', p_field)]
        examples = [] 
        for line in data_lines:
            grapheme, phoneme = line.split(maxsplit=1)
            examples.append(data.Example.fromlist([grapheme, phoneme],
                                                  fields))
        self.sort_key = lambda x: len(x.grapheme)
        super(CMUDict, self).__init__(examples, fields)

    @classmethod
    def splits(cls, path, g_field, p_field, seed=None):
        import random

        if seed is not None:
            random.seed(seed)
        with open(path) as f:
            lines = f.readlines()
        random.shuffle(lines)
        train_lines, val_lines, test_lines = [], [], []
        for i, line in enumerate(lines):
            if i % 20 == 0:
                val_lines.append(line)
            elif i % 20 < 3:
                test_lines.append(line)
            else:
                train_lines.append(line)
        train_data = cls(train_lines, g_field, p_field)
        val_data = cls(val_lines, g_field, p_field)
        test_data = cls(test_lines, g_field, p_field)
        return (train_data, val_data, test_data)
    
filepath = os.path.join(args.data_path, 'cmudict.dict')
train_data, val_data, test_data = CMUDict.splits(filepath, g_field, p_field, args.seed)

g_field.build_vocab(train_data, val_data, test_data)
p_field.build_vocab(train_data, val_data, test_data)

device = None if args.cuda else -1  # None is current gpu
train_iter = data.BucketIterator(train_data, batch_size=args.batch_size,
                                 repeat=False, device='cpu')
val_iter = data.Iterator(val_data, batch_size=1,
                         train=False, sort=False, device='cpu')
test_iter = data.Iterator(test_data, batch_size=1,
                          train=False, shuffle=True, device='cpu')

class Attention(nn.Module):
    """Dot global attention from https://arxiv.org/abs/1508.04025"""
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(dim*2, dim, bias=False)

    def forward(self, x, context: Optional[torch.Tensor]):
        if context is None:
            return x
        assert x.size(0) == context.size(0)  # x: batch x dim
        assert x.size(1) == context.size(2)  # context: batch x seq x dim
        attn = F.softmax(torch.bmm(context, x.unsqueeze(2)).squeeze(2), dim=1)
        weighted_context = torch.bmm(attn.unsqueeze(1), context).squeeze(1)
        o = self.linear(torch.cat((x, weighted_context), 1))
        return F.tanh(o)

class Decoder(nn.Module):

    def __init__(self, vocab_size, d_embed, d_hidden):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.lstm = nn.LSTMCell(d_embed, d_hidden)
        self.attn = Attention(d_hidden)
        self.linear = nn.Linear(d_hidden, vocab_size)

    def forward(self, x_seq, h, c, context: Optional[torch.Tensor]):
        o = []
        e_seq = self.embedding(x_seq)
        for e in torch.chunk(e_seq, e_seq.size(0).item(), 0):
            e = e.squeeze(0)
            h, c = self.lstm(e, (h, c))
            o.append(self.attn(h, context))
        o = torch.stack(o, 0)
        o = self.linear(o.view(-1, h.size(1)))
        return torch.log(F.softmax(o, dim=1)).view(x_seq.size(0), -1, o.size(1)), h, c
    
def train(config, train_iter, model, criterion, optimizer, epoch):
    global iteration, n_total, train_loss, n_bad_loss
    global init, best_val_loss, stop

    print("=> EPOCH {}".format(epoch))
    train_iter.init_epoch()
    for batch in train_iter:
        batch.phoneme = batch.phoneme.to(device)
        batch.grapheme = batch.grapheme.to(device)
        iteration += 1
        model.train()

        output, _, __ = model(batch.grapheme, batch.phoneme[:-1].detach())
        target = batch.phoneme[1:]
        loss = criterion(output.view(output.size(0) * output.size(1), -1),
                         target.view(target.size(0) * target.size(1)))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), config.clip, 'inf')
        optimizer.step()

        n_total += batch.batch_size
        train_loss += loss.item() * batch.batch_size

        if iteration % config.log_every == 0:
            train_loss /= n_total
            val_loss = validate(val_iter, model, criterion)
            print("   % Time: {:5.0f} | Iteration: {:5} | Batch: {:4}/{}"
                  " | Train loss: {:.4f} | Val loss: {:.4f}"
                  .format(time.time()-init, iteration, train_iter.iterations,
                          len(train_iter), train_loss, val_loss))

            # test for val_loss improvement
            n_total = train_loss = 0
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                n_bad_loss = 0
                torch.save(model.state_dict(), config.best_model)
            else:
                n_bad_loss += 1
            if n_bad_loss == config.n_bad_loss:
                best_val_loss = val_loss
                n_bad_loss = 0
                adjust_learning_rate(optimizer, config.lr_decay)
                new_lr = optimizer.param_groups[0]['lr']
                print("=> Adjust learning rate to: {}".format(new_lr))
                if new_lr < config.lr_min:
                    stop = True
                    break

def show(batch, model):
    batch.phoneme = batch.phoneme.to(device)
    batch.grapheme = batch.grapheme.to(device)
    assert batch.batch_size == 1
    g_field = batch.dataset.fields['grapheme']
    p_field = batch.dataset.fields['phoneme']
    prediction = model(batch.grapheme).data.tolist()[:-1]
    grapheme = batch.grapheme.squeeze(1).data.tolist()[1:][::-1]
    phoneme = batch.phoneme.squeeze(1).data.tolist()[1:-1]
    print("> {}\n= {}\n< {}\n".format(
        ''.join([g_field.vocab.itos[g] for g in grapheme]),
        ' '.join([p_field.vocab.itos[p] for p in phoneme]),
        ' '.join([p_field.vocab.itos[p] for p in prediction])))

class G2P(nn.Module):

    def __init__(self, config):
        super(G2P, self).__init__()
        self.encoder = Encoder(config['g_size'], config['d_embed'],
                               config['d_hidden'])
        self.decoder = Decoder(config['p_size'], config['d_embed'],
                               config['d_hidden'])
        for i in config:
          setattr(self, i+'x', config[i])

        self.done = False
        self.pad = 1
        self.bos = 2
        self.eos = 3
        self.tt = torch
        self.scores = self.tt.FloatTensor(self.beam_sizex).zero_()
        self.prevKs = [torch.tensor([0, 0, 0])] #this random tensor is added because torchscipt does not like empty list
        self.pop_prevKs = 1 # this attribute is added to create an if statement in the advance function to remove the random tensor above
        self.nextYs = [self.tt.LongTensor(self.beam_sizex).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

    def forward(self, g_seq):
        context, h, c = self.encoder(g_seq)
        assert g_seq.size(1) == 1  # make sure batch_size = 1
        return self._generate(h, c, context)


    def _generate(self, h, c, context):
        # Make a beam_size batch.
        self.done = False
        self.pad = 1
        self.bos = 2
        self.eos = 3
        self.scores = torch.zeros(self.beam_sizex)
        self.prevKs = [torch.tensor([0, 0, 0])] #this random tensor is added because torchscipt does not like empty list
        self.pop_prevKs = 1 # this attribute is added to create an if statement in the advance function to remove the random tensor above
        self.nextYs = [torch.full([self.beam_sizex],self.pad)]
        self.nextYs[0][0] = self.bos
        h = h.expand(self.beam_sizex, h.size(1))
        c = c.expand(self.beam_sizex, c.size(1))
        context = context.expand(self.beam_sizex, context.size(1), context.size(2)) if self.attentionx else None


        for i in range(self.max_lenx):
            x = self.get_current_state()
            o, h, c = self.decoder(x.unsqueeze(0), h, c, context)
            if self.advance(o.data.squeeze(0)):
                break
            h.data.copy_(torch.index_select(h.data, 0, self.get_current_origin())) 
            c.data.copy_(torch.index_select(c.data, 0, self.get_current_origin())) 
        return torch.stack([self.get_hyp(j) for j in range(self.beam_sizex)])

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if self.pop_prevKs == 1:
          self.prevKs.pop(0)
          self.pop_prevKs = 0
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.beam_sizex, 0,
                                                     True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId // num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True
        return self.done

    def get_hyp(self, k: int):
        hyp = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
        return torch.stack(hyp[::-1])

    def get_current_state(self):
        return self.nextYs[-1]

    def get_current_origin(self):
        return self.prevKs[-1]

config = args
config.g_size = len(g_field.vocab)
config.p_size = len(p_field.vocab)
config.best_model = "3_13_2023.pth"

model = G2P(config.__dict__)
criterion = nn.NLLLoss()
if config.cuda:
    device = torch.device("cpu")
    model.to(device)
optimizer = optim.Adagrad(model.parameters(), lr=config.lr)  # use Adagrad
model.load_state_dict(torch.load("3_13_2023.pth", map_location=torch.device('cpu')))

input_word_token_list = [[2]]
input_word = 'ttttttttttttttttttttttttttttttt'
input_word = input_word[::-1] #input needs to be reversed
for i in input_word:
  input_word_token_list.append([g_field.vocab.itos.index(i)])
input_tensor = torch.Tensor(input_word_token_list).int()

class Bidirectional:
    def __init__(self, layer, backward_layer):
        self.layer = layer
        self.backward_layer = backward_layer

    def __call__(self, x, initial_state=None):
        if initial_state is not None:
            half = len(initial_state) // 2
            state_f = initial_state[:half]
            state_b = initial_state[half:]
        else:
            state_f = None
            state_b = None

        ret_f = self.layer(x, state_f)
        ret_b = self.backward_layer(x, state_b)
        y_f, h_f = ret_f[0], ret_f[1:]
        y_b, h_b = ret_b[0], ret_b[1:]
        y_b = tf.reverse(y_b, axis=(1,))
        y_cat = tf.concat([y_f, y_b], axis=-1)
        return y_cat, *h_f, *h_b

@nobuco.converter(torch.index_select, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def index_select(input: torch.Tensor, dim: int, index: torch.Tensor):
    def func(input, dim, index):
        return tf.gather(input, index, axis=dim)
    return func

@nobuco.converter(nn.LSTMCell, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_LSTM(self: nn.LSTMCell, input, hx=None):
    bidirectional = False
    num_layers = 1

    def create_layer(i, reverse):
        suffix = '_reverse' if reverse else ''
        weight_ih = self.__getattr__(f'weight_ih').cpu().detach().numpy().transpose((1, 0))
        weight_hh = self.__getattr__(f'weight_hh').cpu().detach().numpy().transpose((1, 0))
        weights = [weight_ih, weight_hh]

        if self.bias:
            bias_ih = self.__getattr__(f'bias_ih').cpu().detach().numpy()
            bias_hh = self.__getattr__(f'bias_hh').cpu().detach().numpy()         
            weights += [bias_ih + bias_hh]

        #output needs to give top 10 to match pytorch
        lstm = keras.layers.LSTM(
            units=self.hidden_size,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=self.bias,
            dropout=0,
            return_sequences=True,
            return_state=True,
            time_major=False,
            unroll=False,
            go_backwards=reverse,
            weights=weights,
        )
        return lstm

    def convert_initial_states(hx):
        if hx is None: 
            return None
        else:
            h0, c0 = tuple(tf.reshape(h, (num_layers, -1, *tf.shape(h)[1:])) for h in hx) #may need to be adjusted
            initial_states = []
            for i in range(num_layers):
                if bidirectional:
                    state = (h0[i][0], c0[i][0], h0[i][1], c0[i][1])
                else:
                    state = (h0[i][0], c0[i][0])
                initial_states.append(state)
            return initial_states

    layers = []
    for i in range(num_layers):
        layer = create_layer(i, reverse=False)
        if bidirectional:
            layer_reverse = create_layer(i, reverse=True)
            layer = Bidirectional(layer=layer, backward_layer=layer_reverse)
        layers.append(layer)

    no_batch = input.dim() == 2

    def func(input, hx=None):
        x = input

        if no_batch:
            x = x[:, None, :]
            if hx is not None:
                hxs, cxs = hx
                hxs = hxs[None, :, :]
                cxs = cxs[None, :, :]
                hx = (hxs, cxs)

        initial_states = convert_initial_states(hx)

        hxs = []
        cxs = []
        for i in range(num_layers):
            x, *rec_o = layers[i](x, initial_state=initial_states[i] if initial_states else None) #compare the inputs here to the decoder inputs
            hxs.append(rec_o[0::2])
            cxs.append(rec_o[1::2])
        hxs = tf.concat(hxs, axis=0)
        cxs = tf.concat(cxs, axis=0)

        if no_batch:
            x = x[0, :, :]
            hxs = hxs[0, :, :]
            cxs = cxs[0, :, :]

        return (hxs, cxs)
    return func


keras_model = nobuco.pytorch_to_keras(
    model.eval(),
    args=[input_tensor], kwargs=None,
    input_shapes={input_tensor: (None, 1)}, 
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
    trace_shape=True,
)


np_tensor = input_tensor.numpy()
tf_tensor = tf.convert_to_tensor(np_tensor)

answer = keras_model.predict(tf_tensor)
print(answer)

keras_model.save("./tensorflowjs_model_32_max", save_format='tf')