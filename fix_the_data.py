
import re
import cPickle as pickle
from pprint import pprint
from collections import Counter

def create_data(fpath1, fpath2):
    ret     = {}
    with open(fpath1) as f:
        m = 0
        for l in f.readlines():
            if(m>0):
                t = l.strip().split('||')
                ret[t[0]]           = {}
                text                = t[1].decode('utf-8')
                text                = bioclean(text)
                text                = re.sub('\d', 'D', text)
                ret[t[0]]['text']   = text
            m+=1
        f.close()
    with open(fpath2) as f:
        m = 0
        for l in f.readlines():
            if(m>0):
                t = l.strip().split(',')
                ret[t[0]]['gene']      = t[1].decode('utf-8')
                ret[t[0]]['variation'] = t[2].decode('utf-8')
                if(len(t)>3):
                    ret[t[0]]['class'] = int(t[3])
            m+=1
        f.close()
    return ret

def get_the_vocab(data, min_freq):
    vocab = Counter()
    for item in data.values():
        vocab.update(Counter(item['text'].split()))
    vocab = Counter(dict([item for item in vocab.items() if (item[1] >= min_freq)]))
    return vocab

def get_the_cahrs(vocab):
    chars = []
    for k in vocab.keys():
        chars.extend(k)
    chars = sorted(list(set(chars)))
    return chars

def get_ids(vocab, chars):
    vocab_ids           = dict(zip( sorted(vocab.keys()) , range(2,len(vocab)+2) ))
    vocab_ids['PAD']    = 0
    vocab_ids['UNKN']   = 1
    #
    char_ids            = dict(zip( sorted(chars) , range(2,len(vocab)+2) ))
    char_ids['PAD']    = 0
    char_ids['UNKN']   = 1
    return vocab_ids, char_ids

def get_token_id(vocab_ids, token):
    try:
        return vocab_ids[token]
    except:
        return vocab_ids['UNKN']

bioclean = lambda t: ' '.join(re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split())

datadir     = '/home/dpappas/.kaggle/competitions/msk-redefining-cancer-treatment/'
fpath1      = datadir + 'training_text'
fpath2      = datadir + 'training_variants'
train_data  = create_data(fpath1, fpath2)
fpath1      = datadir + 'test_text'
fpath2      = datadir + 'test_variants'
test_data   = create_data(fpath1, fpath2)
vocab       = get_the_vocab(train_data, 10)
print(len(vocab))
# pprint(vocab.most_common(10))
# pprint(list(reversed(vocab.most_common()[-10:])))
chars   = get_the_cahrs(vocab)
print len(chars)
print(''.join(chars))

vocab_ids, char_ids = get_ids(vocab, chars)
# pprint(vocab_ids)
# pprint(char_ids)

pickle.dump(vocab_ids,  open('vocab_ids.p','wb'))
pickle.dump(char_ids,   open('char_ids.p','wb'))
pickle.dump(test_data,  open('test_data.p','wb'))
pickle.dump(train_data, open('train_data.p','wb'))


# for key in test_data:
#     text                        = test_data[key]['text']
#     test_data[key]['token_ids'] = [ get_token_id(vocab_ids, token) for token in text.split() ]
#     test_data[key]['char_ids']  = [ [ get_token_id(char_ids, c) for c in token ] for token in text.split() ]
#
# for key in train_data:
#     text                            = train_data[key]['text']
#     train_data[key]['token_ids']    = [ get_token_id(vocab_ids, token) for token in text.split() ]
#     train_data[key]['char_ids']     = [ [ get_token_id(char_ids, c) for c in token ] for token in text.split() ]
#
# pprint(train_data.items()[0])





