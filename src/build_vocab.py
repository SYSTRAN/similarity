#!/usr/bin/python -u

import sys
from collections import defaultdict

def main(args):
    Freq = defaultdict(int)
    nlines = 0
    nwords = 0

    input = sys.stdin
    if len(args) >= 2:
        input = open(args[1],"r")
    for line in input:
        nlines += 1
        for word in line.rstrip().split():
            nwords += 1
            Freq[word] += 1

    output = sys.stdout
    if len(args) >= 3:
        output = open(args[2],"w")
    sys.stderr.write("#lines={} #words={} vocab={}\n".format(nlines, nwords, len(Freq)))

    max_vocab = None
    if len(args) >= 4:
        max_vocab = int(args[3])
        sys.stderr.write("Top %s token will be kept.\n" % max_vocab)

    i = 0
    for wrd, frq in sorted(Freq.items(), key=lambda(k, v): v, reverse=True):
        i += 1
        if max_vocab is not None and i > max_vocab:
            break
        output.write("{}\n".format(wrd))

    if input is not sys.stdin:
        input.close()
    if output is not sys.stdout:
        output.close()

if __name__ == "__main__":
    main(sys.argv)
