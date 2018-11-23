#!/usr/bin/python -u

import sys
from collections import defaultdict

Freq = defaultdict(int)
nlines = 0
nwords = 0
for line in sys.stdin:
    nlines += 1
    for word in line.rstrip().split():
        nwords += 1
        Freq[word] += 1

sys.stderr.write("#lines={} #words={} vocab={}\n".format(nlines,nwords,len(Freq)))
for wrd,frq in sorted(Freq.items(), key=lambda(k,v): v, reverse=True): 
    print("{}".format(wrd))




