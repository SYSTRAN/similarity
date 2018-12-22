# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import sys
import os
import time
from random import randint
from config import Config
from dataset import minibatches


class Visualize():

    def __init__(self, n_sents, src, tgt, sim):
        # ... ,aggr_src,aggr_tgt,alig):
        self.n_sents = n_sents
        self.src = src
        self.tgt = tgt
        self.sim = sim
        # self.aggr_src = aggr_src
        # self.aggr_tgt = aggr_tgt
        # self.align = align

    def print_matrix(self, aggr_src, aggr_tgt, align):
        print('<:::{}:::> cosine sim = {:.4f}'.format(self.n_sents, self.sim))
        source = list(self.src)
        target = list(self.tgt)
        for s in range(len(source)):
            if aggr_src[s] < 0:
                source[s] = '*'+source[s]
        for t in range(len(target)):
            if aggr_tgt[t] < 0:
                target[t] = '*'+target[t]

        max_length_tgt_tokens = max(5, max([len(x) for x in target]))
        A = str(max_length_tgt_tokens+1)
        print(''.join(("{:"+A+"}").format(t) for t in target))
        for s in range(len(source)):
            for t in range(len(target)):
                myscore = "{:+.2f}".format(align[s][t])
                while len(myscore) < max_length_tgt_tokens+1:
                    myscore += ' '
                sys.stdout.write(myscore)
            print(source[s])

    def print_svg(self, aggr_src, aggr_tgt, align):
        start_x = 25
        start_y = 100
        len_square = 15
        len_x = len(self.tgt)
        len_y = len(self.src)
        separation = 2
        print "<br>\n<svg width=\""+str(len_x*len_square + start_x + 150)+"\" height=\"" + \
            str(len_y*len_square + start_y)+"\">"
        for x in range(len(self.tgt)):
            if aggr_tgt[x] < 0:
                col = "red"
            else:
                col = "black"
            print "<text x=\""+str(x*len_square + start_x)+"\" y=\""+str(start_y-2)+"\" fill=\""+col + \
                "\" font-family=\"Courier\" font-size=\"5\">"+"{:+.1f}".format(aggr_tgt[x])+"</text>"
            ### remove this line if you want divergent words in red
            col = "black"
            print "<text x=\""+str(x*len_square + start_x + separation)+"\" y=\""+str(start_y-15) + \
                "\" fill=\""+col+"\" font-family=\"Courier\" font-size=\"10\" transform=\"rotate(-45 " + \
                str(x*len_square + start_x + 10)+","+str(start_y-15)+") \">"+self.tgt[x]+"</text>"
        for y in range(len(self.src)):
            for x in range(len(self.tgt)):
                color = align[y][x]
                if color < 0:
                    color = 1
                elif color > 10:
                    color = 0
                else:
                    color = (-color+10)/10
                color = int(color*256)
                print "<rect x=\""+str(x*len_square + start_x)+"\" y=\""+str(y*len_square + start_y) + \
                    "\" width=\""+str(len_square)+"\" height=\""+str(len_square)+"\" style=\"fill:rgb("+str(color) + \
                    ","+str(color)+","+str(color)+"); stroke-width:1;stroke:rgb(200,200,200)\" />"
                txtcolor = "black"
                if align[y][x] < 0:
                    txtcolor = "red"
                print "<text x=\""+str(x*len_square + start_x)+"\" y=\"" + \
                    str(y*len_square + start_y + len_square*3/4) + \
                    "\" fill=\"{}\" font-family=\"Courier\" font-size=\"5\">".format(txtcolor) + \
                    "{:+.1f}".format(align[y][x])+"</text>"

            if aggr_src[y] < 0:
                ### last column with source words
                col = "red"
            else:
                co = "black"
            print "<text x=\""+str(len_x*len_square + start_x + separation)+"\" y=\"" + \
                str(y*len_square + start_y + len_square*3/4)+"\" fill=\""+col + \
                "\" font-family=\"Courier\" font-size=\"5\">"+"{:+.1f}".format(aggr_src[y])+"</text>"
            ### remove this line if you want divergent words in red
            col = "black"
            print "<text x=\""+str(len_x*len_square + start_x + separation + 15)+"\" y=\"" + \
                str(y*len_square + start_y + len_square*3/4)+"\" fill=\""+col + \
                "\" font-family=\"Courier\" font-size=\"10\">"+self.src[y]+"</text>"
        print("<br>\n<svg width=\"200\" height=\"20\">")
        print("<text x=\"{}\" y=\"10\" fill=\"black\" font-family=\"Courier\" font-size=\"8\"\">{:+.4f}</text>".format(
            start_x, self.sim))

    def print_vectors(self, last_src, last_tgt, aggr_src, aggr_tgt, align):
        line = []
        line.append("{:.4f}".format(self.sim))
        line.append(" ".join(s for s in self.src))
        line.append(" ".join(t for t in self.tgt))

        if len(last_src) and len(last_tgt):
            line.append(" ".join("{:.4f}".format(s) for s in last_src))
            line.append(" ".join("{:.4f}".format(t) for t in last_tgt))

        if len(aggr_src) and len(aggr_tgt):
            line.append(" ".join("{:.4f}".format(s) for s in aggr_src))
            line.append(" ".join("{:.4f}".format(t) for t in aggr_tgt))

        if len(align):
            matrix = []
            for s in range(len(self.src)):
                row = " ".join("{:.4f}".format(align[s, t]) for t in range(len(self.tgt)))
                matrix.append(row)
            line.append("\t".join(row for row in matrix))

        print "\t".join(line)
