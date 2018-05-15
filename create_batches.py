
import cPickle as pickle
from pprint import pprint
import os
import re

def get_ids(text):
    text                = ' '.join(text)
    text                = bioclean(text)
    text                = re.sub('\d', 'D', text)
    ret = []
    for token in text.split():
        try:
            ret.append(vocab_ids[token])
        except:
            ret.append(vocab_ids['UNKN'])
    return ret

def pad_sent_ids(sent_ids, max_len):
    ret = []
    for item in sent_ids:
        ret.append(item+ ( (max_len - len(item)) * [0]) )
    return ret

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
items       = test_data.items()
for i in range(0, len(items), b_size):
    pprint(items[i])
    exit()


for item in items:
    pprint(item)
    sent_ids = [ get_ids(sent) for sent in item[1]['text'] ]
    sent_ids = pad_sent_ids(sent_ids, 100)
    pprint(sent_ids)
    exit()







