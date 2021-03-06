
import cPickle as pickle
from pprint import pprint
import numpy as np
import os
import re

def get_ids(text):
    text    = bioclean(text)
    text    = re.sub('\d', 'D', text)
    ret     = []
    for token in text.split():
        try:
            ret.append(vocab_ids[token])
        except:
            ret.append(vocab_ids['UNKN'])
    return ret

def pad_sent_ids(sent_ids, max_sent_len, max_len):
    ret = []
    for item in sent_ids:
        p = (max_sent_len - len(item)) * [0]
        ret.append(item + p )
    ps = [0] * max_sent_len
    ps = (max_len - len(ret)) * [ps]
    ret.extend(ps)
    return ret

def batch_from_data(items):
    max_len         = max(len(item['text']) for item in items)
    max_sent_len    = max(max(len(s.split()) for s in item['text'] ) for item in items)
    print max_sent_len
    genes, targets, variations, sent_ids = [], [], [], []
    for item in items:
        targets.append(item['class'])
        genes.append(gene_ids[item['gene']])
        variations.append(variation_ids[item['variation']])
        sids = [ get_ids(s) for s in item['text']]
        sids = pad_sent_ids(sids, max_sent_len, max_len)
        sent_ids.append(sids)
    targets     = np.array(targets)
    genes       = np.array(genes)
    variations  = np.array(variations)
    sent_ids    = np.array(sent_ids)
    print targets.shape
    print genes.shape
    print variations.shape
    print sent_ids.shape
    exit()


bioclean = lambda t: ' '.join(re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split())

odir        = './batches/'
train_dir   = odir+'train/'
test_dir    = odir+'test/'

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

variation_ids   = pickle.load(open('variation_ids.p','rb'))
gene_ids        = pickle.load(open('gene_ids.p','rb'))
vocab_ids       = pickle.load(open('vocab_ids.p','rb'))
test_data       = pickle.load(open('test_data.p','rb'))
train_data      = pickle.load(open('train_data.p','rb'))

b_size      = 64
items       = train_data.values()
for i in range(0, len(items), b_size):
    batch = batch_from_data(items[i:min([i+b_size, len(items)])])
    pprint(batch)
    exit()


for item in items:
    pprint(item)
    sent_ids = [ get_ids(sent) for sent in item[1]['text'] ]
    sent_ids = pad_sent_ids(sent_ids, 100)
    pprint(sent_ids)
    exit()







