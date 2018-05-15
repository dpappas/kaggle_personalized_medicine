
import re
import cPickle as pickle
from pprint import pprint
from collections import Counter
from nltk.tokenize import sent_tokenize

def create_data(fpath1, fpath2):
    ret     = {}
    with open(fpath1) as f:
        m = 0
        for l in f.readlines():
            if(m>0):
                t = l.strip().split('||')
                ret[t[0]]           = {}
                ret[t[0]]['text']   = sent_tokenize(t[1].strip().decode('utf-8'))
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
        text                = ' '.join(item['text'])
        text                = bioclean(text)
        text                = re.sub('\d', 'D', text)
        vocab.update(Counter(text.split()))
    vocab = Counter(dict([item for item in vocab.items() if (item[1] >= min_freq)]))
    return vocab

def get_the_chars(vocab):
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

def first_alpha_is_upper(sent):
    specials = [
        '__EU__','__SU__','__EMS__','__SMS__','__SI__',
        '__ESB','__SSB__','__EB__','__SB__','__EI__',
        '__EA__','__SA__','__SQ__','__EQ__','__EXTLINK',
        '__XREF','__URI', '__EMAIL','__ARRAY','__TABLE',
        '__FIG','__AWID','__FUNDS'
    ]
    for special in specials:
        sent = sent.replace(special,'')
    for c in sent:
        if(c.isalpha()):
            if(c.isupper()):
                return True
            else:
                return False
    return False

def ends_with_special(sent):
    sent = sent.lower()
    ind = [item.end() for item in re.finditer('[\W\s]sp.|[\W\s]nos.|[\W\s]figs.|[\W\s]sp.[\W\s]no.|[\W\s][vols.|[\W\s]cv.|[\W\s]fig.|[\W\s]e.g.|[\W\s]et[\W\s]al.|[\W\s]i.e.|[\W\s]p.p.m.|[\W\s]cf.|[\W\s]n.a.|[\W\s]no.', sent)]
    if(len(ind)==0):
        return False
    else:
        ind = max(ind)
        if (len(sent) == ind):
            return True
        else:
            return False

def split_sentences2(text):
    sents = [l.strip() for l in sent_tokenize(text)]
    ret = []
    i = 0
    while (i < len(sents)):
        sent = sents[i]
        while (
            ((i + 1) < len(sents)) and
            (
                ends_with_special(sent)        or
                not first_alpha_is_upper(sents[i+1])
                # sent[-5:].count('.') > 1       or
                # sents[i+1][:10].count('.')>1   or
                # len(sent.split()) < 2          or
                # len(sents[i+1].split()) < 2
            )
        ):
            sent += ' ' + sents[i + 1]
            i += 1
        ret.append(sent.replace('\n',' ').strip())
        i += 1
    return ret

def get_sents(ntext):
    sents = []
    for subtext in ntext.split('\n'):
        subtext = re.sub( '\s+', ' ', subtext.replace('\n',' ') ).strip()
        if (len(subtext) > 0):
            ss = split_sentences2(subtext)
            sents.extend([ s for s in ss if(len(s.strip())>0)])
    if(len(sents[-1]) == 0 ):
        sents = sents[:-1]
    return sents

bioclean = lambda t: ' '.join(re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split())

datadir     = '/home/dpappas/.kaggle/competitions/msk-redefining-cancer-treatment/'
fpath1      = datadir + 'training_text'
fpath2      = datadir + 'training_variants'
train_data  = create_data(fpath1, fpath2)
fpath1      = datadir + 'test_text'
fpath2      = datadir + 'test_variants'
test_data   = create_data(fpath1, fpath2)
vocab       = get_the_vocab(train_data, 10)
chars       = get_the_chars(vocab)
vocab_ids, char_ids = get_ids(vocab, chars)
print(len(vocab))
# pprint(vocab.most_common(10))
# pprint(list(reversed(vocab.most_common()[-10:])))
# print len(chars)
# print(''.join(chars))
# pprint(vocab_ids)
# pprint(char_ids)

pprint(test_data.items()[0])

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





