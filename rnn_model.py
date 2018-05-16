
my_seed = 1989
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import cPickle as pickle
import numpy as np
import random
random.seed(my_seed)
import os
import sys
import time
import shutil
from nltk import ngrams
from pprint import pprint
from sklearn.metrics import roc_curve, auc
cudnn.benchmark = True
torch.manual_seed(my_seed)
print torch.get_num_threads()
print torch.cuda.is_available()
print torch.cuda.device_count()

use_cuda = torch.cuda.is_available()
if(use_cuda):
    torch.cuda.manual_seed(my_seed)

gpu_device = 0

class IR_Embeddings_Modeler(nn.Module):
    def __init__(
            self,
            hidden_dim_sentence,
            out_size,
            vocab_size,
            embedding_dim,
            pool_method         = 'attention',
            dropout_prob        = 0.2
    ):
        super(IR_Embeddings_Modeler, self).__init__()
        self.hidden_dim_sent                        = hidden_dim_sentence
        self.out_size                               = out_size
        self.pooling                                = pool_method
        self.vocab_size                             = vocab_size
        self.embedding_dim                          = embedding_dim
        #
        self.word_embeddings                        = nn.Embedding(self.vocab_size, self.embedding_dim)
        #
        self.sent_h         = torch.nn.Parameter(torch.randn(2, 1, self.hidden_dim_sent))
        self.sentence_rnn   = nn.GRU(
            input_size      = self.embedding_dim,
            hidden_size     = self.hidden_dim_sent,
            num_layers      = 1,
            bidirectional   = True,
            bias            = True,
            dropout         = 0,
            batch_first     = True
        )
        self.out_dense      = nn.Linear(2 * self.hidden_dim_quest + 2 * self.hidden_dim_sent, self.out_size, bias=True)
        if(use_cuda):
            self.word_embeddings    = self.word_embeddings.cuda(gpu_device)
            self.out_dense          = self.out_dense.cuda(gpu_device)
    def get_last(self, matrix, lengths):
        ret = [ matrix[i,lengths[i]-1] for i in range(matrix.size(0)) ]
        return torch.stack(ret)
    def fix_input(self, sentence, target):
        sentence        = autograd.Variable(torch.LongTensor(sentence), requires_grad=False)
        target          = autograd.Variable(torch.LongTensor(target), requires_grad=False)
        #
        sentence_len    = [torch.nonzero(item).size(0) for item in sentence.data]
        #
        max_s_len       = torch.max(autograd.Variable(torch.LongTensor(sentence_len)))
        #
        sentence        = sentence[:, :max_s_len.data[0]]
        #
        if(use_cuda):
            sentence = sentence.cuda(gpu_device)
            target = target.cuda(gpu_device)
        return sentence, target, sentence_len
    def pool_that_thing(self, the_thing, the_thing_lens):
        if(self.pooling     == 'max'):
            ret, max_idxs    = torch.max(the_thing, 1)
            ret              = ret.squeeze(1)
        elif(self.pooling   == 'last'):
            ret              = self.get_last(the_thing, the_thing_lens)
        else:
            ret              = the_thing.sum(1) / the_thing.size(1)
            ret              = ret.squeeze(1)
        return ret
    def forward(self, sentence, target):
        sentence, target, sentence_len = self.fix_input(sentence, target)
        #
        sent_embeds             = self.word_embeddings(sentence)
        #
        sent_h                  = torch.cat(sent_embeds.size(0) * [self.sent_h], dim=1)
        sentence_rnn_out, hn1   = self.sentence_rnn(sent_embeds, sent_h)
        sentence_last           = self.pool_that_thing(sentence_rnn_out, sentence_len)
        print(sentence_last.size())
        exit()
        #
        if(self.output_layer_method == 'mlp'):
            concatenated            = torch.cat([sentence_last, ngram_last], dim=-1)
            d1 = self.dense1(concatenated)
            d1 = F.sigmoid(d1)
            d2 = self.dense2(d1)
            losss = F.cross_entropy(d2, target, weight=None, size_average=True)
            d2 = F.softmax(d2)
        else:
            sentence_rnn_out    = F.relu(sentence_rnn_out)
            ngram_rnn_out       = F.relu(ngram_rnn_out)
            ngram_last          = F.relu(ngram_last)
            sentence_last       = F.relu(sentence_last)
            sim                 = self.cos(ngram_last, sentence_last)
            sim             = torch.clamp(sim, min=0., max=1.0)
            # print(sim.cpu().data.numpy())
            losss               = F.binary_cross_entropy(sim, target.float())
            d2                  = sim
        #
        # return losss, F.softmax(d2,-1), sentence_last, ngram_last, sentence_rnn_out, ngram_rnn_out
        return losss, d2, sentence_last, ngram_last, sentence_rnn_out, ngram_rnn_out


model = IR_Embeddings_Modeler(
    hidden_dim_sentence = 100,
    out_size            = 100,
    vocab_size          = 100,
    embedding_dim       = 100,
    pool_method         = 'max'
)

lr          = 0.001
optimizer   = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)








