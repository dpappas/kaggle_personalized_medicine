
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
# use_cuda = False
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
        self.out_dense      = nn.Linear( 2 * self.hidden_dim_sent, self.out_size, bias=True)
        self.loss_fun       = nn.NLLLoss()
        if(use_cuda):
            self.word_embeddings    = self.word_embeddings.cuda(gpu_device)
            self.sentence_rnn       = self.sentence_rnn.cuda(gpu_device)
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
        # if(use_cuda):
        #     sent_embeds = sent_embeds.cuda(gpu_device)
        #
        sent_h                  = torch.cat(sent_embeds.size(0) * [self.sent_h], dim=1)
        sentence_rnn_out, hn1   = self.sentence_rnn(sent_embeds, sent_h)
        sentence_last           = self.pool_that_thing(sentence_rnn_out, sentence_len)
        output                  = self.out_dense(sentence_last)
        output                  = F.sigmoid(output)
        output                  = F.softmax(output, -1)
        losss                   = F.cross_entropy(output, target, weight=None, size_average=True)
        # losss = self.loss_fun(output, target)
        # print(sent_embeds.size())
        # print(sentence_last.size())
        # print(output.size())
        # print(losss.size())
        return losss

model = IR_Embeddings_Modeler(
    hidden_dim_sentence = 100,
    out_size            = 9,
    vocab_size          = 43501,
    embedding_dim       = 50,
    pool_method         = 'last'
)

if(use_cuda):
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda(gpu_device)

lr          = 0.1
optimizer   = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

diri    = './batches/train/'
fs      = os.listdir(diri)
print(len(fs))

for e in range(10):
    for i in range(len(fs)-5):
        d = pickle.load(open(diri + fs[i], 'rb'))
        optimizer.zero_grad()
        cost_       = model(d['sent_ids'][:,:100], d['targets']-1)
        cost_.backward()
        optimizer.step()
        the_cost    = cost_.cpu().data.numpy()[0]
        print e, i, len(fs)-5, the_cost

