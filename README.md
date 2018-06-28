# similarity
Bilingual sentence similarity classifier based on word alignments optimization using Tensorflow

This repo implements a sentence similarity classifier model using Tensorflow. Similarity classification is based on the ideas introduced by [Carpuat et al., 2017](http://cs.umd.edu/~yogarshi/publications/2017/wnmtacl2017.pdf) and [Vyas et al., 2018](https://arxiv.org/abs/1803.11112). The code borrows many of the concepts and architecture presented in [Legrand et al., 2016](https://arxiv.org/pdf/1606.09560).

The next picture, shows an example of similarity classification for a parallel sentence pair:

``` Hi , Maggie . Have you heard anything ? ||| As-tu entendu quelque chose ?```

<img src="https://github.com/jmcrego/divergence/blob/master/pics/divergence_example.png" width="250" />