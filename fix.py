# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import string
import math
import sys
import os
import time
from random import randint

class Align():
    def __init__(self,align):
        self.align = align
        self.lsrc = len(align)
        self.ltgt = len(align[0])
        self.max_cost_s = [[[None for c in range(self.ltgt)] for b in range(self.ltgt)] for a in range(self.lsrc)] # max_cost_s[0][1][2] means max cost of source word 0 from target word 1 to target word 2
        self.max_cost_t = [[[None for c in range(self.lsrc)] for b in range(self.lsrc)] for a in range(self.ltgt)]

    def max_s(self,s,t_ini,t_end):
        if self.max_cost_s[s][t_ini][t_end] is not None: 
            cmax = self.max_cost_s[s][t_ini][t_end]
        elif t_ini == t_end:
            cmax = self.align[s][t_ini]
            self.max_cost_s[s][t_ini][t_end] = cmax
        else:
            cmax = max(self.max_s(s,t_ini,t_end-1) , self.max_s(s,t_end,t_end))
            self.max_cost_s[s][t_ini][t_end] = cmax
        return cmax

    def max_t(self,t,s_ini,s_end):
        if self.max_cost_t[t][s_ini][s_end] is not None: 
            cmax = self.max_cost_t[t][s_ini][s_end]
        elif s_ini == s_end:
            cmax = self.align[s_ini][t]
            self.max_cost_t[t][s_ini][s_end] = cmax
        else:
            cmax = max(self.max_t(t,s_ini,s_end-1) , self.max_t(t,s_end,s_end)) 
            self.max_cost_t[t][s_ini][s_end] = cmax
        return cmax

    def cost_square_max_outside(self,s_ini,s_end,t_ini,t_end):
        c = 0.0
        ### source words
        for s in range(0,s_ini):           c -= self.max_s(s,0,self.ltgt-1)
        for s in range(s_end+1,self.lsrc): c -= self.max_s(s,0,self.ltgt-1)
        ### target words
        for t in range(0,t_ini):           c -= self.max_t(t,0,self.lsrc-1)
        for t in range(t_end+1,self.ltgt): c -= self.max_t(t,0,self.lsrc-1)
        return c

    def cost_square_max_inside(self,s_ini,s_end,t_ini,t_end):
        c = 0.0
        ### source words
        for s in range(s_ini,s_end+1): c += self.max_s(s,t_ini,t_end)
        ### target words
        for t in range(t_ini,t_end+1): c += self.max_t(t,s_ini,s_end)
        return c

class Fix():
    def __init__(self,tau,nbest,max_sim):
        self.tau = tau #hypotheses must be at least sized of tau
        self.nbest = nbest
        self.max_sim = max_sim #only hypotheses with a similarity score lower than max_sim are considered to be fixed
        self.inside = True
        self.outside = False
        self.use_punct = True
#        self.ini_punct = {'.', ',', ';', ')', '(' ']', '[', '?', '!', "\'", '\"', '-'}
#        self.end_punct = {'.', ',', ';', ')', '(' ']', '[', '?', '!', "\'", '\"', '-'}


    def print_fix_square(self, src, tgt, align, sim, n_sents):
        a = Align(align)
        min_length = self.tau
        cost = {}
        #print("min_length={}".format(min_length))

        ref_pair = " ".join(s for s in src) + "\t" + " ".join(t for t in tgt) ## original sentence pair
        cost[ref_pair] = a.cost_square_max_inside(0,len(src)-1,0,len(tgt)-1) 
        ### first hyp is always the original
        print("{}\t{:.4f}\t{:.4f}\t{}".format(n_sents,sim,cost[ref_pair],ref_pair))
        #max_cost = cost[ref_pair]

        if sim <= self.max_sim:
            n_times = 0
            for s_ini in range(0,len(src)):
#                if a.max_s(s_ini,0,len(tgt)-1) < 0: continue
                for s_end in range(s_ini+min_length-1,len(src)):
#                    if a.max_s(s_end,0,len(tgt)-1) < 0: continue
                    if not self.bounded_by_punctuation(s_ini,s_end,src): continue
                    for t_ini in range(0,len(tgt)):
#                        if a.max_t(t_ini,s_ini,s_end) < 0: continue
                        for t_end in range(t_ini+min_length-1,len(tgt)):
                            if not self.bounded_by_punctuation(t_ini,t_end,tgt): continue
#                            if a.max_t(t_end,s_ini,s_end) < 0: continue
#                            if a.max_s(s_ini,t_ini,t_end) < 0: continue
#                            if a.max_s(s_end,t_ini,t_end) < 0: continue
                            c = 0.0
                            if self.inside:  c += a.cost_square_max_inside(s_ini,s_end,t_ini,t_end) 
                            if self.outside: c += a.cost_square_max_outside(s_ini,s_end,t_ini,t_end)
                            pair = " ".join(src[s] for s in range(s_ini,s_end+1)) + "\t" + " ".join(tgt[t] for t in range(t_ini,t_end+1))
                            cost[pair] = c
                            #print("{}\t[{},{}][{},{}]\t{:.3f}\t{}".format(n_sents,s_ini,s_end,t_ini,t_end,c,pair))
                            n_times += 1

        #print("min_length={} n_times={}".format(min_length,n_times))
        ### the rest of hyps are sorted by cost
        cost_sorted = sorted(cost, key=cost.get, reverse=True)
        n = 0
        for pair in cost_sorted:
            if pair != ref_pair: 
                #if cost[pair] <= max_cost: break
                print("{}\t{:.4f}\t{:.4f}\t{}".format(n_sents,sim,cost[pair],pair))
                n += 1
                if n >= self.nbest: break

    def bounded_by_punctuation(self,ini,end,vec):
        if not self.use_punct: return True
        if ini > 0 and vec[ini-1] not in string.punctuation: return False ### previous to ini is punctuation
        if vec[end] not in string.punctuation: return False ### end is punctuation
        return True

def main():

    tau = 2
    nbest = 15
    max_sim = 0.1
    ratio=3
    name = sys.argv.pop(0)
    usage = "usage: " + name + " [-tau INT] [-nbest INT] [-max_sim FLOAT]\n"
    while len(sys.argv):
        tok = sys.argv.pop(0)
        if   (tok=="-tau" and len(sys.argv)):     tau = int(sys.argv.pop(0))
        elif (tok=="-nbest" and len(sys.argv)):   nbest = int(sys.argv.pop(0))
        elif (tok=="-max_sim" and len(sys.argv)): max_sim = float(sys.argv.pop(0))
        elif (tok=="-h"):
            sys.stderr.write("{}".format(usage))
            sys.exit()
        else:
            sys.stderr.write('error: unparsed {} option\n'.format(tok))
            sys.stderr.write("{}".format(usage))
            sys.exit()

    fix = Fix(tau,nbest,max_sim)
    n_sent = 0
    for line in sys.stdin:
        n_sent += 1
        tokens = line.strip().split('\t')
        sim = float(tokens.pop(0))
        #print(sim)
        src = tokens.pop(0).split(' ')
        #print(src)
        tgt = tokens.pop(0).split(' ')
        #print(tgt)
        align = []
        while len(tokens) > 0:
            align_tgt = map(float, tokens.pop(0).split(' '))
            align.append(align_tgt)
        #print(align)
        fix.print_fix_square(src,tgt,align,sim,n_sent)

if __name__ == "__main__":
    main()
