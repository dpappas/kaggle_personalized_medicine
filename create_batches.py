
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

def pad_ids(ids_list, max_len):
    ret = []
    for item in ids_list:
        p = (max_len - len(item)) * [0]
        ret.append(item + p)
    return ret

def batch_from_data(items):
    genes, targets, variations, sent_ids = [], [], [], []
    for item in items:
        try:
            targets.append(item['class'])
        except:
            targets.append(0.)
        genes.append(gene_ids[item['gene']])
        variations.append(variation_ids[item['variation']])
        sids = get_ids(item['text'])
        sent_ids.append(sids)
    #
    max_len  = max(len(sid) for sid in sent_ids)
    sent_ids = pad_ids(sent_ids, max_len)
    #
    targets     = np.array(targets)
    genes       = np.array(genes)
    variations  = np.array(variations)
    sent_ids    = np.array(sent_ids)
    return {
        'targets'       : targets,
        'genes'         : genes,
        'variations'    : variations,
        'sent_ids'      : sent_ids
    }

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
    pickle.dump(batch, open(train_dir+'{}.p'.format(i),'wb'))
    print i, len(items)

items       = test_data.values()
for i in range(0, len(items), b_size):
    batch = batch_from_data(items[i:min([i+b_size, len(items)])])
    pickle.dump(batch, open(test_dir+'{}.p'.format(i),'wb'))
    print i, len(items)





