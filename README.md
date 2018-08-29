# similarity
Bilingual sentence similarity classifier based on word alignments optimization using Tensorflow

This repo implements a sentence similarity classifier model using Tensorflow. Similarity classification is based on the ideas introduced by [Carpuat et al., 2017](http://aclweb.org/anthology/W17-3209) similar to [Vyas et al., 2018](http://aclweb.org/anthology/N18-1136), [Schwenk, 2018](http://aclweb.org/anthology/P18-2037) and [Grégoire et al., 2018](http://www.aclweb.org/anthology/C18-1122). The code borrows many of the concepts and architecture presented in [Legrand et al., 2016](http://www.aclweb.org/anthology/W16-2207). Further details obaut the current implementation and experiments are published in [Pham et al., 2018]() at EMNLP2018.

The next picture, shows an example of similarity classification for a parallel sentence pair:

``` What do you feel ? Not . ||| Que ressentez-vous ?```

<img src="https://github.com/jmcrego/similarity/blob/master/divergence_example.png" width="250" />
