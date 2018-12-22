# -*- coding: utf-8 -*-

import numpy as np
import os
import io
import sys
import time
from random import shuffle
from collections import defaultdict

reload(sys)
sys.setdefaultencoding('utf8')


class options():

    def __init__(self, argv):
        self.seq_size = 50
        self.max_sents = 0
        self.seed = 1234
        self.mode = "p"
        self.shuffle = False
        self.debug = False
        self.replace = None
        self.data = None
        usage = """usage: {}
   -seq_size       INT : sentences larger than this number of src/tgt words are filtered out [50]
   -max_sents      INT : Consider this number of sentences per batch (0 for all) [0]
   -seed           INT : seed for randomness [1234]
   -shuffle            : shuffle data
   -debug              : debug mode
   -h                  : this help

*  -data          FILE : training data
   -mode        STRING : how data examples are generated (p: parallel, u:uneven, i:insert, r:replace d:delete) [p]
   -replace       FILE : equivalent sequences (needed when -data_mode contains r)

- Options marked with * must be set. The rest have default values.
""".format(argv.pop(0))

        while len(argv):
            tok = argv.pop(0)
            if (tok == "-data" and len(argv)):
                self.data = argv.pop(0)
            elif (tok == "-max_sents" and len(argv)):
                self.max_sents = int(argv.pop(0))
            elif (tok == "-seq_size" and len(argv)):
                self.seq_size = int(argv.pop(0))
            elif (tok == "-mode" and len(argv)):
                self.mode = argv.pop(0)
            elif (tok == "-debug"):
                self.debug = True
            elif (tok == "-seed" and len(argv)):
                self.seed = int(argv.pop(0))
            elif (tok == "-replace" and len(argv)):
                self.replace = argv.pop(0)
            elif (tok == "-h"):
                sys.stderr.write("{}".format(usage))
                sys.exit()
            else:
                sys.stderr.write('error: unparsed {} option\n'.format(tok))
                sys.stderr.write("{}".format(usage))
                sys.exit()

        if self.data is None:
            sys.stderr.write('error: missing -data optioon\n{}'.format(usage))
            sys.exit()


class stats():

    def __init__(self):
        self.n_parallel = 0
        self.n_uneven = 0
        self.n_insert = 0
        self.n_replace = 0
        self.n_delete = 0
        self.n_src_words = 0
        self.n_tgt_words = 0
        self.n_sents = 0
        self.n_src_divergent = 0
        self.n_tgt_divergent = 0

    def show(self, t):
        sys.stderr.write("Built sentences: {} time: {:.3f} s\n".format(self.n_sents, t))
        if self.n_parallel:
            sys.stderr.write("\t{} parallel ({:.2f}%)\n".format(self.n_parallel, 100.0*self.n_parallel/self.n_sents))
        if self.n_uneven:
            sys.stderr.write("\t{} uneven ({:.2f}%)\n".format(self.n_uneven, 100.0*self.n_uneven/self.n_sents))
        if self.n_insert:
            sys.stderr.write("\t{} insert ({:.2f}%)\n".format(self.n_insert, 100.0*self.n_insert/self.n_sents))
        if self.n_replace:
            sys.stderr.write("\t{} replace ({:.2f}%)\n".format(self.n_replace, 100.0*self.n_replace/self.n_sents))
        if self.n_delete:
            sys.stderr.write("\t{} delete ({:.2f}%)\n".format(self.n_delete, 100.0*self.n_delete/self.n_sents))
        sys.stderr.write("words: {}/{} divergent: {}/{} ({:.2f}%/{:.2f}%)\n".format(
            self.n_src_words, self.n_tgt_words, self.n_src_divergent, self.n_tgt_divergent,
            100.0*self.n_src_divergent/self.n_src_words, 100.0*self.n_tgt_divergent/self.n_tgt_words))


class align():

    def __init__(self, src, tgt, ali):
        self.s2t_minmax = [[None, None] for i in range(len(src))]
        self.t2s_minmax = [[None, None] for i in range(len(tgt))]
        for a in ali:
            if len(a.split('-')) != 2:
                sys.stderr.write('warning: bad ali {}\n'.format(ali))
                continue
            s, t = map(int, a.split('-'))
            if s >= len(src) or t >= len(tgt):
                sys.stderr.write('warning: ali {}-{} out of bounds\nsrc: {}\ntgt: {}\nali: {}\n'.format(
                    s, t, src, tgt, ali))
                continue
            if self.s2t_minmax[s][0] is None:
                self.s2t_minmax[s][0] = t
                self.s2t_minmax[s][1] = t
            else:
                if t < self.s2t_minmax[s][0]:
                    self.s2t_minmax[s][0] = t
                if t > self.s2t_minmax[s][1]:
                    self.s2t_minmax[s][1] = t
            if self.t2s_minmax[t][0] is None:
                self.t2s_minmax[t][0] = s
                self.t2s_minmax[t][1] = s
            else:
                if s < self.t2s_minmax[t][0]:
                    self.t2s_minmax[t][0] = s
                if s > self.t2s_minmax[t][1]:
                    self.t2s_minmax[t][1] = s


class replace():

    def __init__(self, file):
        self.max_length = -1
        self.min_length = -1
        self.pos_to_wrd = defaultdict(list)
        if file is None:
            return
        t0 = time.time()
        with io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            nline = 0
            for line in f:
                nline += 1
                if nline == 1:
                    tok = line.split(",")
                    self.min_length = int(tok[0])
                    self.max_length = int(tok[1])
                    continue
                seqpos, seqwrd = line.strip().split('\t')
                self.pos_to_wrd[seqpos].append(seqwrd)

        t1 = time.time()
        sys.stderr.write('Read replace ({} pos entries) time: {:.3f} s\n'.format(len(self.pos_to_wrd), t1-t0))

    def get(self, seq, pos):
        seqpos = " ".join(pos)
        if "-" in seqpos:
            return []
        seqwrd = " ".join(seq)
        # print("\tseqpos: {}".format(seqpos))
        if seqpos not in self.pos_to_wrd:
            return []
        # is a list of strings
        seqwrds = self.pos_to_wrd[seqpos]

        indexs = [i for i in range(min(100, len(seqwrds)))]
        shuffle(indexs)
        for i in indexs:
            seqwrd2 = seqwrds[i]
            seq2 = seqwrd2.split(' ')
            totally_different = True
            for k in range(len(seq2)):
                if seq2[k] == seq[k]:
                    totally_different = False
                    break
            if totally_different:
                return seq2

        return []


class dataset():

    def __init__(self, r):
        self.SRC = []
        self.TGT = []
        self.ALI = []
        self.POS = []
        self.max_ntry = 50
        self.r = r

    def __len__(self):
        return len(self.SRC)

    def add(self, src, tgt, ali, pos):
        if len(src):
            self.SRC.append(src)
        if len(tgt):
            self.TGT.append(tgt)
        if len(ali):
            self.ALI.append(ali)
        if len(pos):
            self.POS.append(pos)

    def parallel_pair(self, i, st, o):
        src = list(self.SRC[i])
        tgt = list(self.TGT[i])
        src_tag = ['-1.0' for i in range(len(src))]
        tgt_tag = ['-1.0' for i in range(len(tgt))]
        print("{}\t{}\t{}\t{}".format(" ".join(src), " ".join(tgt), " ".join(src_tag), " ".join(tgt_tag)))
        st.n_parallel += 1
        st.n_src_words += len(src)
        st.n_tgt_words += len(tgt)
        st.n_sents += 1
        if o.debug:
            self.debug('PARALLEL', src, src_tag, tgt, tgt_tag)

    def uneven_pair(self, i, st, o):
        n_try = 0
        while True:
            n_try += 1
            if n_try > self.max_ntry:
                ### cannot find the right pair
                return
            j = np.random.randint(0, len(self.SRC))
            if j == i:
                continue
            if i % 2 == 0:
                ### replace src
                src = list(self.SRC[j])
                tgt = list(self.TGT[i])
            else:
                ### replace tgt
                src = list(self.SRC[i])
                tgt = list(self.TGT[j])
            if len(src) > len(tgt) and len(src)*0.5 > len(tgt):
                continue
            if len(tgt) > len(src) and len(tgt)*0.5 > len(src):
                continue
            break

        src_tag = ['1.0' for i in range(len(src))]
        tgt_tag = ['1.0' for i in range(len(tgt))]
        print("{}\t{}\t{}\t{}".format(" ".join(src), " ".join(tgt), " ".join(src_tag), " ".join(tgt_tag)))
        st.n_uneven += 1
        st.n_src_words += len(src)
        st.n_tgt_words += len(tgt)
        st.n_src_divergent += len(src)
        st.n_tgt_divergent += len(tgt)
        st.n_sents += 1
        if o.debug:
            self.debug('UNEVEN (ntry={})'.format(n_try), src, src_tag, tgt, tgt_tag)

    def insert_pair(self, i, st, o):
        src = list(self.SRC[i])
        tgt = list(self.TGT[i])
        src_tag = ['-1.0' for i in range(len(src))]
        tgt_tag = ['-1.0' for i in range(len(tgt))]

        where = ""
        if len(src) <= len(tgt):
            ### add in src side
            n_try = 0
            while True:
                n_try += 1
                if n_try > self.max_ntry:
                    ### cannot find the right pair
                    return
                j = np.random.randint(0, len(self.SRC))
                if j == i:
                    continue
                ### replace src
                add = list(self.SRC[j])
                new_src = len(src)+len(add)
                if new_src > len(tgt) and new_src > len(tgt)*2:
                    continue
                if len(tgt) > new_src and len(tgt) > new_src*2:
                    continue
                break
            if i % 2 == 0:
                ### add in the begining
                where = "src:begin"
                for k in range(len(add)):
                    src.insert(0, add[len(add)-k-1])
                    src_tag.insert(0, '1.0')
            else:
                ### add in the end
                where = "src:end"
                for k in range(len(add)):
                    src.append(add[k])
                    src_tag.append('1.0')
            st.n_src_divergent += len(add)
        else:
            ### add in tgt side
            n_try = 0
            while True:
                n_try += 1
                if n_try > self.max_ntry:
                    ### cannot find the right pair
                    return
                j = np.random.randint(0, len(self.SRC))
                if j == i:
                    continue
                ### replace tgt
                add = list(self.TGT[j])
                new_tgt = len(tgt)+len(add)
                if len(src) > new_tgt and len(src) > new_tgt*2:
                    continue
                if new_tgt > len(src) and new_tgt > len(src)*2:
                    continue
                break
            if i % 2 == 0:
                ### add in the begining
                where = "tgt:begin"
                for k in range(len(add)):
                    tgt.insert(0, add[len(add)-k-1])
                    tgt_tag.insert(0, '1.0')
            else:
                ### add in the end
                where = "tgt:end"
                for k in range(len(add)):
                    tgt.append(add[k])
                    tgt_tag.append('1.0')
            st.n_tgt_divergent += len(add)

        print("{}\t{}\t{}\t{}".format(" ".join(src), " ".join(tgt), " ".join(src_tag), " ".join(tgt_tag)))
        st.n_insert += 1
        st.n_src_words += len(src)
        st.n_tgt_words += len(tgt)
        st.n_sents += 1
        if o.debug:
            self.debug('INSERT {} (ntry={})'.format(where, n_try), src, src_tag, tgt, tgt_tag)
        return

    def delete_pair(self, i, st, o):
        if len(self.ALI) == 0:
            return
        src_orig = list(self.SRC[i])
        tgt_orig = list(self.TGT[i])
        ali = list(self.ALI[i])
        a = align(src_orig, tgt_orig, ali)
        n_try = 0
        while True:
            n_try += 1
            if n_try >= self.max_ntry:
                return
            imin = np.random.randint(0, len(tgt_orig))
            imax = np.random.randint(imin, len(tgt_orig))
            if imax-imin+1 > len(tgt_orig)//2:
                continue
            src_from, src_to, tgt_from, tgt_to = self.expand_self_contained_sequence(imin, imax, a)
            if src_from < 0 or src_to < src_from or src_to >= len(src_orig):
                continue
            if tgt_from < 0 or tgt_to < tgt_from or tgt_to >= len(tgt_orig):
                continue
            if src_to-src_from+1 > len(src_orig)//2:
                continue
            if tgt_to-tgt_from+1 > len(tgt_orig)//2:
                continue
            break

        src = []
        src_tag = []
        tgt = []
        tgt_tag = []
        deleted = []
        where = ""
        if len(src_orig) >= len(tgt_orig):
            ### delete on src side mark on tgt side
            where = "src"
            for s in range(len(src_orig)):
                if s < src_from or s > src_to:
                    ### keep the word
                    src.append(src_orig[s])
                    src_tag.append('-1.0')
                else:
                    deleted.append(src_orig[s])
            for t in range(len(tgt_orig)):
                ### keep all tgt words
                tgt.append(tgt_orig[t])
                if t < tgt_from or t > tgt_to:
                    ### mark non-divergent (the aligned src word is kept)
                    tgt_tag.append('-1.0')
                else:
                    ### mark divergent (the aligned src word is deleted)
                    tgt_tag.append('1.0')
                    st.n_tgt_divergent += 1
        else:
            ### delete on tgt side mark on src side
            where = "tgt"
            for t in range(len(tgt_orig)):
                if t < tgt_from or t > tgt_to:
                    ### keep the word
                    tgt.append(tgt_orig[t])
                    tgt_tag.append('-1.0')
                else:
                    deleted.append(tgt_orig[t])
            for s in range(len(src_orig)):
                ### keep all src words
                src.append(src_orig[s])
                if s < src_from or s > src_to:
                    ### mark non-divergent (the aligned tgt word is kept)
                    src_tag.append('-1.0')
                else:
                    ### mark divergent (the aligned tgt word is deleted)
                    src_tag.append('1.0')
                    st.n_src_divergent += 1

        print("{}\t{}\t{}\t{}".format(" ".join(src), " ".join(tgt), " ".join(src_tag), " ".join(tgt_tag)))
        st.n_delete += 1
        st.n_src_words += len(src)
        st.n_tgt_words += len(tgt)
        st.n_sents += 1
        if o.debug:
            self.debug('DELETE {} [{},{}][{},{}] {} (ntry={})'.format(
                    where, src_from, src_to, tgt_from, tgt_to, " ".join(deleted), n_try), src, src_tag, tgt, tgt_tag)
        return

    def replace_pair(self, i, st, o):
        if self.r.max_length == -1:
            return
        if len(self.ALI) == 0:
            return
        if len(self.POS) == 0:
            return
        src = list(self.SRC[i])
        tgt = list(self.TGT[i])
        ali = list(self.ALI[i])
        pos = list(self.POS[i])
        if len(src) < 3:
            return

        src_from, src_to, tgt_from, tgt_to, replace_by = self.find_replacement(src, tgt, ali, pos)
        if len(replace_by) == 0:
            return
        if src_from < 0 or src_to < src_from or src_to >= len(src):
            return
        if src_from == 0 and src_to == len(src)-1:
            return

        replace_orig = " ".join([src[s] for s in range(src_from, src_to+1)])
        replace_dest = " ".join(replace_by)
        src_tag = ['-1.0' for i in range(len(src))]
        tgt_tag = ['-1.0' for i in range(len(tgt))]
        ###
        ### replace words in source side
        ###
        for s in range(src_from, src_to+1):
            src[s] = replace_by[s-src_from]
            src_tag[s] = '1.0'
            st.n_src_divergent += 1
        ###
        ### keep words in target side except those aligned with the replaced in the source side
        ###
        for t in range(tgt_from, tgt_to+1):
            tgt_tag[t] = '1.0'
            st.n_tgt_divergent += 1

        print("{}\t{}\t{}\t{}".format(" ".join(src), " ".join(tgt), " ".join(src_tag), " ".join(tgt_tag)))
        st.n_replace += 1
        st.n_src_words += len(src)
        st.n_tgt_words += len(tgt)
        st.n_sents += 1
        if o.debug:
            self.debug('REPLACE [{},{}][{},{}] {} => {}'.format(
                src_from, src_to, tgt_from, tgt_to, replace_orig, replace_dest), src, src_tag, tgt, tgt_tag)
        return

    def debug(self, tag, src, src_tag, tgt, tgt_tag):
        print("{}".format(tag))
        print("\tsrc     : [{}] {}".format(len(src), " ".join([s for s in src])))
        print("\tsrc_tag : [{}] {}".format(len(src_tag), " ".join([str(s) for s in src_tag])))
        print("\ttgt     : [{}] {}".format(len(tgt), " ".join([t for t in tgt])))
        print("\ttgt_tag : [{}] {}".format(len(tgt_tag), " ".join([str(t) for t in tgt_tag])))

    def find_replacement(self, src, tgt, ali, pos):
        if len(src) < self.r.min_length*2:
            return -1, -1, -1, -1, []
        if len(ali) == 0:
            ### there must be at leat one alignment
            return -1, -1, -1, -1, []
        if len(pos) != len(src):
            return -1, -1, -1, -1, []

        a = align(src, tgt, ali)
        # build alignment structures for fast access
        # s2t_minmax = [[None, None] for i in range(len(src))]
        # t2s_minmax = [[None, None] for i in range(len(tgt))]
        # for a in ali:
        #    s, t = map(int, a.split('-'))
        #    if s2t_minmax[s][0] is None:
        #        s2t_minmax[s][0] = t
        #        s2t_minmax[s][1] = t
        #    else:
        #        if t < s2t_minmax[s][0]: s2t_minmax[s][0] = t
        #        if t > s2t_minmax[s][1]: s2t_minmax[s][1] = t
        #    if t2s_minmax[t][0] is None:
        #        t2s_minmax[t][0] = s
        #        t2s_minmax[t][1] = s
        #    else:
        #        if s < t2s_minmax[t][0]: t2s_minmax[t][0] = s
        #        if s > t2s_minmax[t][1]: t2s_minmax[t][1] = s
        # 0 : build alignment structures for fast access

        # find the largest replacement that does not exceeds len(src)/2
        replace_by = []
        src_from = -1
        src_to = -1
        tgt_from = -1
        tgt_to = -1
        indexs = [i for i in range(len(src))]
        shuffle(indexs)
        for src_from in indexs:
            # for src_to in range(src_from+self.r.min_length-1,src_from+self.r.max_length):
            for src_to in range(src_from+self.r.max_length-1, src_from+self.r.min_length-2, -1):
                if src_to >= len(src):
                    continue
                if src_to-src_from+1 > len(src)//2:
                    continue
                tgt_from, tgt_to, replace_by = self.replacement(src, tgt, src_from, src_to, pos, a)
                if len(replace_by) > 0:
                    return src_from, src_to, tgt_from, tgt_to, replace_by
        return src_from, src_to, tgt_from, tgt_to, replace_by

    def replacement(self, src, tgt, src_from, src_to, pos, a):
        src_seq = src[src_from:src_to+1]
        pos_seq = pos[src_from:src_to+1]
        src_rep = self.r.get(src_seq, pos_seq)
        if len(src_rep) != len(src_seq):
            return -1, -1, []

        tgt_from = 9999
        tgt_to = -1
        ### first i collect the [tgt_from,tgt_to] words aligned to [src_from, src_to]
        for s in range(src_from, src_to+1):
            tgt_min = a.s2t_minmax[s][0]
            tgt_max = a.s2t_minmax[s][1]
            if tgt_min is not None and tgt_min < tgt_from:
                tgt_from = tgt_min
            if tgt_max is not None and tgt_max > tgt_to:
                gt_to = tgt_max
        if tgt_from == 9999 or tgt_to == -1:
            return -1, -1, []

        ### second i make sure that any target word in [tgt_from, tgt_to] is not aligned outside [src_from,src_to]
        for t in range(tgt_from, tgt_to+1):
            src_min = a.t2s_minmax[t][0]
            src_max = a.t2s_minmax[t][1]
            if src_min is not None and src_min < src_from:
                return -1, -1, []
            if src_max is not None and src_max > src_to:
                return -1, -1, []
        return tgt_from, tgt_to, src_rep

    def expand_self_contained_sequence(self, imin, imax, a):
        jmin = None
        jmax = None
        length_sequences = (imax-imin+1)
        # 2 : find the minimum sequence of source/target words that contains [jmin,jmax][imin,imax] and are
        # not aligned outside the sequence
        j_or_i = 0
        while True:
            if j_or_i == 0:
                jmin, jmax = self.MinMax(a.t2s_minmax, imin, imax)
            else:
                imin, imax = self.MinMax(a.s2t_minmax, jmin, jmax)
            if jmin is None or jmax is None or imin is None or imax is None:
                return -1, -1, -1, -1
            if length_sequences == (imax-imin+1) + (jmax-jmin+1):
                # no added words stop iterating
                break
            length_sequences = (imax-imin+1) + (jmax-jmin+1)
            j_or_i = (j_or_i + 1) % 2
        # extend with unaligned words ???
        # filter bad examples: all words, too many tokens, etc.
        # max_ratio = 0.5 #maximum ratio of deleted words
        # if jmin == 0 and jmax == len(src)-1: return -1, -1, -1, -1
        # if imin == 0 and imax == len(tgt)-1: return -1, -1, -1, -1
        # if ((imax-imin+1) + (jmax-jmin+1))/(len(src)+len(tgt)) > max_ratio: return -1, -1, -1, -1
        return jmin, jmax, imin, imax

    def MinMax(self, a2b_minmax, amin, amax):
        bmin = None
        bmax = None
        for a in range(amin, amax+1):
            if a2b_minmax[a][0] is not None:
                if bmin is None or a2b_minmax[a][0] < bmin:
                    bmin = a2b_minmax[a][0]
            if a2b_minmax[a][1] is not None:
                if bmax is None or a2b_minmax[a][1] > bmax:
                    bmax = a2b_minmax[a][1]
        return bmin, bmax


def main():
    o = options(sys.argv)
    r = replace(o.replace)
    d = dataset(r)
    t0 = time.time()
    with io.open(o.data, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        i = 0
        n_total = 0
        n_src_prune = 0
        n_tgt_prune = 0
        for line in f:
            n_total += 1
            if n_total % 100000 == 0:
                if n_total % 1000000 == 0:
                    sys.stderr.write(str(n_total))
                else:
                    sys.stderr.write(".")
            i += 1
            tok = line.strip('\n').split("\t")
            if len(tok) < 2:
                sys.stderr.write('warning: line {} contains less than 2 fields [skipping line]\n'.format(i))
                continue
            if len(tok) > 4:
                sys.stderr.write('warning: line {} contains more than 4 fields [skipping line]\n'.format(i))
                continue
            src = tok.pop(0).strip().split(' ')
            tgt = tok.pop(0).strip().split(' ')
            ali = []
            pos = []
            if o.seq_size > 0 and len(src) > o.seq_size:
                n_src_prune += 1
                continue
            if o.seq_size > 0 and len(tgt) > o.seq_size:
                n_tgt_prune += 1
                continue
            if len(tok) > 0:
                ali = tok.pop(0).strip().split(' ')
                if len(ali) == 0:
                    sys.stderr.write('warning: empty alignments in line {} [skipping line]\n'.format(i))
                    continue
            if len(tok) > 0:
                pos = tok.pop(0).strip().split(' ')
                if len(pos) > 0 and len(src) != len(pos):
                    sys.stderr.write('warning: different number of src/pos tokens in line %d [skipping line]\n' % i)
                    continue
            d.add(src, tgt, ali, pos)
    t1 = time.time()
    sys.stderr.write('Read {} sentence pairs ({}/{} pruned) time: {:.3f} s\n'.format(
        n_total, n_src_prune, n_tgt_prune, t1-t0))

    st = stats()
    indexs = [i for i in range(len(d))]
    if o.shuffle:
        shuffle(indexs)
    for i in indexs:
        if 'p' in o.mode:
            d.parallel_pair(i, st, o)
        if 'u' in o.mode:
            d.uneven_pair(i, st, o)
        if 'i' in o.mode:
            d.insert_pair(i, st, o)
        if 'r' in o.mode:
            d.replace_pair(i, st, o)
        if 'd' in o.mode:
            d.delete_pair(i, st, o)
        if o.max_sents > 0 and s.n_sents >= o.max_sents:
            break
    st.show(t2-t1)


if __name__ == "__main__":
    main()
