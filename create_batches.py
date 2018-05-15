
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

bioclean = lambda t: ' '.join(re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split())

odir        = './batches/'
train_dir   = odir+'train/'
test_dir    = odir+'test/'

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

vocab_ids   = pickle.load(open('vocab_ids.p','rb'))
test_data   = pickle.load(open('test_data.p','rb'))
train_data  = pickle.load(open('train_data.p','rb'))

for item in test_data.values():
    pprint(item['text'])
    sent_ids = [ get_ids(sent) for sent in item['text'] ]
    pprint(sent_ids)
    exit()







