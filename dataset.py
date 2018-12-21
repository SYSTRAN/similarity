# -*- coding: utf-8 -*-
import os.path
import io
from math import *
from random import shuffle
import numpy as np
import sys
import time
import gzip
from collections import defaultdict
from tokenizer import build_tokenizer

reload(sys)
sys.setdefaultencoding('utf8')

idx_unk = 0
str_unk = "<unk>"

idx_pad = 1
str_pad = "<pad>"

class Embeddings():

    def __init__(self, file, voc, length):
        w2e = {}
        if file is not None:
            #with io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            if file.endswith('.gz'): f = gzip.open(file, 'rb')
            else: f = io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')

            self.num, self.dim = map(int, f.readline().split())
            i = 0
            for line in f:
                i += 1
                if i%10000 == 0:
                    if i%100000 == 0: sys.stderr.write("{}".format(i))
                    else: sys.stderr.write(".")
                tokens = line.rstrip().split(' ')
                if voc.exists(tokens[0]): w2e[tokens[0]] = tokens[1:] 
            f.close()
            sys.stderr.write('Read {} embeddings ({} missing in voc)\n'.format(len(w2e),len(voc)-len(w2e)))
        else:
            sys.stderr.write('Embeddings file not used! will be initialised to [{}x{}]\n'.format(len(voc),length))
            self.dim = length

        # i need an embedding for each word in voc
        # embedding matrix must have tokens in same order than voc 0:<unk>, 1:<pad>, 2:le, ...
        self.matrix = []
        for tok in voc:
            if tok == str_unk or tok == str_pad or not tok in w2e: ### random initialize these tokens
                self.matrix.append(np.random.normal(0, 1.0, self.dim)) 
            else:
                self.matrix.append(np.asarray(w2e[tok], dtype=np.float32))

        self.matrix = np.asarray(self.matrix, dtype=np.float32)
        self.matrix = self.matrix / np.sqrt((self.matrix ** 2).sum(1))[:, None]

class Vocab():

    def __init__(self, dict_file):
        self.tok_to_idx = {}
        self.idx_to_tok = []
        self.idx_to_tok.append(str_unk)
        self.tok_to_idx[str_unk] = len(self.tok_to_idx) #0
        self.idx_to_tok.append(str_pad)
        self.tok_to_idx[str_pad] = len(self.tok_to_idx) #1
        #with io.open(dict_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        nline = 0
        for line in [line.rstrip('\n') for line in open(dict_file)]:
            nline += 1
            line = line.strip()
            self.idx_to_tok.append(line)
            self.tok_to_idx[line] = len(self.tok_to_idx)

        self.length = len(self.idx_to_tok)
        sys.stderr.write('Read vocab ({} entries)\n'.format(self.length))

    def __len__(self):
        return len(self.idx_to_tok)

    def __iter__(self):
        for tok in self.idx_to_tok:
            yield tok

    def exists(self, s):
        return s in self.tok_to_idx

    def get(self,s):
        if type(s) == int: ### I want the string
            if s < len(self.idx_to_tok): return self.idx_to_tok[s]
            else:
                sys.stderr.write('error: key \'{}\' not found in vocab\n'.format(s))
                sys.exit()
        ### I want the index
        if s not in self.tok_to_idx: return idx_unk
        return self.tok_to_idx[s]


class Dataset():

    def __init__(self, file, voc_src, tok_src, voc_tgt, tok_tgt, seq_size, max_sents, do_shuffle):
        if file is None:
            return
        self.voc_src = voc_src
        self.voc_tgt = voc_tgt
        self.file = file
        self.seq_size = seq_size
        self.max_sents = max_sents
        self.do_shuffle = do_shuffle
        self.annotated = False
        self.data = []
        self.length = 0 ### length of the data set to be used (not necessarily the whole set)

        src_tokenizer = None
        tgt_tokenizer = None
        if tok_src:
            src_tokenizer = build_tokenizer(tok_src)
        if tok_tgt:
            tgt_tokenizer = build_tokenizer(tok_tgt)

        if self.file.endswith('.gz'):
            f = gzip.open(self.file, 'rb')
        else:
            f = io.open(self.file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        firstline = True
        for line in f:
            lsplit = line.split('\t')
            if firstline:
                if len(lsplit) == 4:
                    self.annotated = True
                firstline = False
            if src_tokenizer:
                tokens, _ = src_tokenizer.tokenize(str(lsplit[0]))
                lsplit[0] = " ".join(tokens)
            if tgt_tokenizer:
                tokens, _ = tgt_tokenizer.tokenize(str(lsplit[1]))
                lsplit[1] = " ".join(tokens)
            self.data.append("\t".join(lsplit))
            self.length += 1
        f.close()

        if self.max_sents > 0:
            self.length = min(self.length, self.max_sents)
        sys.stderr.write('({} contains {} examples)\n'.format(self.file, len(self.data)))

    def __iter__(self):
        nsent = 0
        self.nsrc = 0
        self.ntgt = 0
        self.nunk_src = 0
        self.nunk_tgt = 0
        self.ndiv_src = 0
        self.ndiv_tgt = 0
        ### every iteration i get shuffled data examples if do_shuffle
        indexs = [i for i in range(len(self.data))]
        if self.do_shuffle:
            shuffle(indexs)
        for index in indexs:
            tokens = self.data[index].strip().split('\t')
            if len(tokens) != 2 and len(tokens) != 4:
                sys.stderr.write("warning: bad data entry \'{}\' in line={} [skipped]\n".format(self.data[index],index+1))
                continue

            src = tokens[0].split(' ')
            tgt = tokens[1].split(' ')
            if self.seq_size > 0 and (len(src) > self.seq_size or len(tgt) > self.seq_size): continue # filter out examples with more than seq_size tokens
            if len(tokens) == 2: ### test set without annotations
                src_tag_txt = ['-1.0' for i in src]
                tgt_tag_txt = ['-1.0' for i in tgt]
            else:
                src_tag_txt = tokens[2].split(' ')
                tgt_tag_txt = tokens[3].split(' ')
                if len(tgt_tag_txt)!=len(tgt) or len(src_tag_txt)!=len(src):
                    sys.stderr.write("warning: diff num of words/tags \'{}\' in line={} [skipped]\n".format(self.data[index],index+1))
                    continue

            isrc, itgt, src_tag, tgt_tag = self.build_example(src,tgt,src_tag_txt,tgt_tag_txt)
            self.keep_records(src_tag, tgt_tag, isrc, itgt)
            yield isrc, itgt, src, tgt, src_tag, tgt_tag
            nsent += 1
            if self.max_sents > 0 and nsent > self.max_sents: break # already generated max_sents examples

    def __len__(self):
        return self.length

    def build_example(self,src,tgt,src_tag_txt,tgt_tag_txt):
        isrc = []
        src_tag = []
        i = 0
        for s in src: 
            isrc.append(self.voc_src.get(s))
            if src_tag_txt[i]=='-1.0': src_tag.append(-1.0)
            else: src_tag.append(1.0)
            i += 1

        itgt = []
        tgt_tag = []
        i = 0
        for t in tgt: 
            itgt.append(self.voc_tgt.get(t))
            if tgt_tag_txt[i]=='-1.0': tgt_tag.append(-1.0)
            else: tgt_tag.append(1.0)
            i += 1

        return isrc, itgt, src_tag, tgt_tag

    def keep_records(self, src_tag, tgt_tag, isrc, itgt):
        self.ndiv_src += sum(1 for s in src_tag if s==1.0) 
        self.ndiv_tgt += sum(1 for t in tgt_tag if t==1.0)
        self.nunk_src += sum(1 for s in isrc if s==idx_unk)
        self.nunk_tgt += sum(1 for t in itgt if t==idx_unk)
        self.nsrc += len(src_tag)
        self.ntgt += len(tgt_tag)

def minibatches(data, minibatch_size):
    SRC, TGT, RAW_SRC, RAW_TGT, SRC_TAG, TGT_TAG = [], [], [], [], [], []
    max_src, max_tgt = 0, 0
    for (src, tgt, raw_src, raw_tgt, src_tag, tgt_tag) in data:
        if len(SRC) == minibatch_size:
            yield build_batch(SRC, TGT, RAW_SRC, RAW_TGT, SRC_TAG, TGT_TAG, max_src, max_tgt)
            SRC, TGT, RAW_SRC, RAW_TGT, SRC_TAG, TGT_TAG = [], [], [], [], [], []
            max_src, max_tgt = 0, 0
        if len(src) > max_src: max_src = len(src)
        if len(tgt) > max_tgt: max_tgt = len(tgt)
        SRC.append(src)
        TGT.append(tgt)
        RAW_SRC.append(raw_src)
        RAW_TGT.append(raw_tgt)
        SRC_TAG.append(src_tag)
        TGT_TAG.append(tgt_tag)

    if len(SRC) != 0:
        yield build_batch(SRC, TGT, RAW_SRC, RAW_TGT, SRC_TAG, TGT_TAG, max_src, max_tgt)

def build_batch(SRC, TGT, RAW_SRC, RAW_TGT, SRC_TAG, TGT_TAG, max_src, max_tgt):
    src_batch, tgt_batch, raw_src_batch, raw_tgt_batch, sign_src_batch, sign_tgt_batch, sign_batch, len_src_batch, len_tgt_batch = [], [], [], [], [], [], [], [], []
    ### build: src_batch, pad_src_batch sized of max_src
    batch_size = len(SRC)
    for i in range(batch_size):
        # add padding to have max_src/max_tgt words in all examples of current batch
        sign_src = SRC_TAG[i]
        sign_tgt = TGT_TAG[i]
        sign = max(max(sign_src),max(sign_tgt)) ### sign is divergent (+1) if there is any divergent word
        while len(sign_src) < max_src: sign_src.append(-1.0) ### continue filling up to max_tgt cells
        while len(sign_tgt) < max_tgt: sign_tgt.append(-1.0) ### continue filling up to max_tgt cells
        src = list(SRC[i])
        tgt = list(TGT[i])
        while len(src) < max_src: src.append(idx_pad) #<pad>
        while len(tgt) < max_tgt: tgt.append(idx_pad) #<pad>
        raw_src = list(RAW_SRC[i])
        raw_tgt = list(RAW_TGT[i])
        len_src = len(SRC[i])
        len_tgt = len(TGT[i])
        ### add to batches
        sign_src_batch.append(sign_src)
        sign_tgt_batch.append(sign_tgt)
        sign_batch.append(sign)
        src_batch.append(src)
        tgt_batch.append(tgt)
        raw_src_batch.append(raw_src)
        raw_tgt_batch.append(raw_tgt)
        len_src_batch.append(len_src)
        len_tgt_batch.append(len_tgt)

    return src_batch, tgt_batch, raw_src_batch, raw_tgt_batch, sign_src_batch, sign_tgt_batch, sign_batch, \
        len_src_batch, len_tgt_batch
