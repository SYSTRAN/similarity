# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import sys
from fix import Fix


def main():

    tau = 2
    nbest = 15
    max_sim = 0.1
    ratio = 3
    name = sys.argv.pop(0)
    usage = "usage: " + name + " [-tau INT] [-nbest INT] [-max_sim FLOAT]\n"
    while len(sys.argv):
        tok = sys.argv.pop(0)
        if (tok == "-tau" and len(sys.argv)):
            tau = int(sys.argv.pop(0))
        elif (tok == "-nbest" and len(sys.argv)):
            nbest = int(sys.argv.pop(0))
        elif (tok == "-max_sim" and len(sys.argv)):
            max_sim = float(sys.argv.pop(0))
        elif (tok == "-h"):
            sys.stderr.write("{}".format(usage))
            sys.exit()
        else:
            sys.stderr.write('error: unparsed {} option\n'.format(tok))
            sys.stderr.write("{}".format(usage))
            sys.exit(1)

    fix = Fix(tau, nbest, max_sim)
    n_sent = 0
    for line in sys.stdin:
        n_sent += 1
        tokens = line.strip().split('\t')
        sim = float(tokens.pop(0))
        # print(sim)
        src = tokens.pop(0).split(' ')
        # print(src)
        tgt = tokens.pop(0).split(' ')
        # print(tgt)
        align = []
        while len(tokens) > 0:
            align_tgt = map(float, tokens.pop(0).split(' '))
            align.append(align_tgt)
        # print(align)
        fix.print_fix_square(src, tgt, align, sim, n_sent)


if __name__ == "__main__":
    main()
