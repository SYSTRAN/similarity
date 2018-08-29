# similarity
Bilingual sentence similarity classifier based on optimising word alignments using Tensorflow.

This repo implements a sentence similarity classifier model using Tensorflow. Similarity classification is based on the ideas introduced by [Carpuat et al., 2017](http://aclweb.org/anthology/W17-3209) and similar to [Vyas et al., 2018](http://aclweb.org/anthology/N18-1136), [Schwenk, 2018](http://aclweb.org/anthology/P18-2037) and [Grégoire et al., 2018](http://www.aclweb.org/anthology/C18-1122). The code borrows many of the concepts and architecture presented in [Legrand et al., 2016](http://www.aclweb.org/anthology/W16-2207). Further details obout the current implementation and experiments are published in [Pham et al., 2018]() at EMNLP2018.

The next picture, shows an example of similarity classification for the sentence pair:

``` What do you feel ? Not . ||| Que ressentez-vous ?```

<img src="https://github.com/jmcrego/similarity/blob/master/divergence_example.png" width="250" />

As it can be seen, the model outputs:
* an overall sentence similarity score (0.1201),
* aggregation scores at the level of words (shown next to words) and
* alignment scores for each pair of words.

# Learning
```
python ./divergence_tagger.py -mdir DIR \
                              -dev FILE \
                              -trn FILE \
                              -wrd_dict FILE \
                              -tag_dict FILE \
                              -emb_size 100 \
                              -seq_size 100 \
                              -lstm_size 64 \
                              -batch_size 4 \
                              -lr_method sgd \
                              -lr 1.0 \
                              -lr_decay 0.8 \
                              -dropout 0.3 \
                              -n_epochs 15
```
# Inference
```
python ./divergence_tagger.py -model FILE \
                              -tst FILE \
                              -wrd_dict FILE \
                              -tag_dict FILE \
                              -emb_size 100 \
                              -seq_size 100 \
                              -lstm_size 64 \
                              -evaluate
```